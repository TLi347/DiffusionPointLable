from cProfile import label
from itertools import cycle
from locale import normalize
import os
import math
import argparse
import torch
import numpy as np
from torch.nn.modules.container import ModuleList
import torch.utils.tensorboard
from torch.utils.data import DataLoader, dataset
from torch.utils.data import ConcatDataset
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import point_cloud_utils as pcu

from utils.dataset import *
from utils.misc import *
from utils.data import *
# from models.vae_gaussian import *
# from models.vae_flow import *
from compare.pointnet.pointnet import *
from compare.pointnet4.pointnet2_part_seg_ssg import *
from compare.dgcnn.model import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from utils import part_dataset_all_normal
from utils.few_shot_part_dataset_all_normal import FewShotPartNormalDataset
from src.datasets import InMemoryPCLabelDataset
os.chdir('/data2/tli/seg-diffusion-point-cloud')

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='pointnetpp', choices=['pointnet', 'pointnetpp', 'dgcnn', 'sgm'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--sample_num_points', type=int, default=2048)

# Datasets and loaders
parser.add_argument('--train_dataset_choice', type=str, default='shapenet',choices=['shapenet', 'ours', 'mixture'])
parser.add_argument('--dataset_path', type=str, default='/data2/tli/normal-auxiliary-diffusion-point-cloud/data/shapenetcore_partanno_segmentation_benchmark_v0_normal')
parser.add_argument('--categories', type=str, default='chair')
parser.add_argument('--label_min', type=int, default=12)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--k_shot', type=int, default=1)
parser.add_argument('--k_sample', type=int, default=256)
parser.add_argument('--uncertainty_portion', type=float,  default=0.3)
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--dataset_normalize', type=bool, default=False)

#sgm ckpt
parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_05_22__20_42_21/ckpt_0.000000_102000.pt')#chair
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_02_18__01_13_27/ckpt_0.000000_240000.pt')#airplane
parser.add_argument('--sgm_steps', type=list, default=[1,5,10,15,20])
parser.add_argument('--sgm_blocks', type=list, default=[0])
parser.add_argument('--sgm_upsample_mode', type=str, default='linear')
parser.add_argument('--input_activations', type=bool, default=False)
parser.add_argument('--start_model_num', type=int, default=0)
parser.add_argument('--model_num', type=int, default=10)
parser.add_argument('--upsample_mode', type=str, default='linear')

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-6)
parser.add_argument('--sched_start_epoch', type=int, default=200*THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400*THOUSAND)

# Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./res_1tab_rep')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_iters', type=int, default=100000)
parser.add_argument('--val_freq', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default='_')
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix=args.categories+'_'+str(args.k_shot)+'_'+args.model+'_', postfix='')
    logger = get_logger('train', log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    ckpt_mgr = BlackHole()
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.to(y.device)
    return new_y


# Model
from compare.src.feature_extractors import create_feature_extractor, collect_features, collect_features_batch
from compare.src.pixel_classifier import pixel_classifier, load_ensemble, predict_labels
import json
rnd_gen = torch.Generator(device=args.device).manual_seed(args.seed)
noise = torch.randn(1, 2048, 3, generator=rnd_gen, device=args.device).repeat(2,1,1)
logger.info('Building model...')
if args.model == 'pointnet':
    model = PointNetDenseCls(k=args.num_classes, feature_transform=False).to(args.device)
elif args.model == 'pointnetpp':
    model = pointnet2_part_seg_ssg(args.num_classes).to(args.device)
elif args.model == 'dgcnn':
    model = DGCNN_partseg(args, args.num_classes).to(args.device)
elif args.model == 'sgm':
    # SGM
    args.features_dim = [256]
    args.dim = [2048,256*5]
    feature_extractor = create_feature_extractor(args)
    model = pixel_classifier(numpy_class=args.num_classes, dim=args.dim[-1])
    model.init_weights()
    model = model.to(device=args.device)
    opts = json.load(open('./logs_exp_dir/ddpm.json', 'r'))
    opts.update(vars(args))
# logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)
# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), 
    lr=args.lr, 
    weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)


# Datasets and loaders
logger.info('Loading Training datasets...')
dataset = FewShotPartNormalDataset(root=args.dataset_path, split='train', normalize=args.dataset_normalize, npoints=2048, cates=args.categories, k_shot=args.k_shot)
dataloader = DataLoader(dataset,batch_size=1,shuffle=True,drop_last=True)
num_batch = len(dataset) // 2#args.train_batch_size
logger.info(('Train shape num: %d') % (len(dataset)))

i = 0
for epoch in range(20):
    print('epoch: ', epoch)
    for data in enumerate(dataloader):
        points, _, target = data[1]
        
        points = points.to(dtype=torch.float32).repeat(2,1,1)
        target = (target).to(dtype=torch.long).repeat(2,1,1)
        print('TRAIN target:  ', target.min().item(), target.max().item())

        
        points, target = points.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        model = model.train()
        if args.model == 'pointnet':
            points = points.transpose(2, 1)
            pred, _, trans_feat = model(points)
        elif args.model == 'pointnetpp':
            points = points.transpose(2, 1)
            label = torch.zeros((points.shape[0],1)).to(device=points.device, dtype=points.dtype)
            pred, _, trans_feat = model(points, to_categorical(label, 16))
        elif args.model == 'dgcnn':
            points = points.transpose(2, 1)
            label = torch.zeros((points.shape[0])).to(device=points.device, dtype=points.dtype)
            pred, _, trans_feat = model(points, to_categorical(label, 16))
        elif args.model == 'sgm':
            X = torch.zeros((args.train_batch_size, *opts['dim']), dtype=torch.float)
            y = torch.zeros((args.train_batch_size, *opts['dim'][:-1]), dtype=torch.uint8)
            pc = (points).to(device=args.device,dtype=torch.float32)
            print('pc', pc.shape)
            features = feature_extractor(pc, noise=noise)
            print('features, ', len(features), features[0].shape)
            X = collect_features_batch(opts, features).cpu()
            print('X,  ', X.shape)
            
            X_batch, y_batch = X.permute(0,2,1).to(args.device), y.to(args.device)
            y_batch = y_batch.type(torch.long)
            pred = model(X_batch)
            y_pred_softmax = torch.log_softmax(pred, dim=1)
            pred = y_pred_softmax.transpose(2,1)
            print(pred.shape)
        pred = pred.contiguous().view(-1, args.num_classes)
        target = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(pred, target)
        # if args.feature_transform:
        #     loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward() 
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        logger.info('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(args.train_batch_size * 2048)))
        i += 1
    torch.save(model.state_dict(), os.path.join(log_dir, '%s_%s.pth' % (args.model, args.categories)))
    scheduler.step()
# ckpt = torch.load(os.path.join(log_dir, '%s_%s.pth' % (args.model, args.categories)))
# model.load_state_dict(ckpt)
model.eval()

# Val on Real
TEST_DATASET1 = FewShotPartNormalDataset(root=args.dataset_path, npoints=2048, split='test', normalize=False, cates=args.categories, k_shot=-1)
testset_min_label = args.label_min
testdataloader1 = DataLoader(TEST_DATASET1, batch_size=16, shuffle=True, drop_last=True)
noise = noise[0:1,:,:].repeat(16,1,1)
shape_ious = []
logger.info('Start evaluating of ShapeNet....')
for i,data in tqdm(enumerate(testdataloader1, 0)):
    points, _, target = data
    # pcu.save_mesh_v(os.path.join(log_dir,'val.ply'), points[0].detach().cpu().numpy())
    points = points.to(dtype=torch.float32)
    target = (target).to(dtype=torch.long)
    print('TEST target:  ', target.min(), target.max())

    points, target = points.to(args.device), target.to(args.device)
    # model = model.eval()
    # pred, _, _ = model(points)
    if args.model == 'pointnet':
        points = points.transpose(2, 1)
        pred, _, _ = model(points)
    elif args.model == 'pointnetpp':
        points = points.transpose(2, 1)
        label = torch.zeros((points.shape[0],1)).to(device=points.device, dtype=points.dtype)
        pred, _, _ = model(points, to_categorical(label, 16))
    elif args.model == 'dgcnn':
        points = points.transpose(2, 1)
        label = torch.zeros((points.shape[0])).to(device=points.device, dtype=points.dtype)
        pred, _, trans_feat = model(points, to_categorical(label, 16))
    elif args.model == 'sgm':
        X = torch.zeros((args.train_batch_size, *opts['dim']), dtype=torch.float)
        pc = (points).to(device=args.device,dtype=torch.float32)
        print('pc', pc.shape)
        features = feature_extractor(pc, noise=noise)
        print('features, ', len(features), features[0].shape)
        X = collect_features_batch(opts, features).cpu()
        print('X,  ', X.shape)
        
        X_batch, y_batch = X.permute(0,2,1).to(args.device), y.to(args.device)
        y_batch = y_batch.type(torch.long)
        pred = model(X_batch)
        y_pred_softmax = torch.log_softmax(pred, dim=1)
        pred = y_pred_softmax.transpose(2,1)
        print(pred.shape)
    pred_choice = pred.data.max(2)[1] #65536,4 -> 65536

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy()# - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(args.num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

# print("mIOU for class {}: {}".format(args.categories, np.mean(shape_ious)))
logger.info("val mIOU for class {}: {}".format(args.categories, np.mean(shape_ious)))