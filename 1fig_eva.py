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
parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'pointnetpp', 'dgcnn'])
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--sample_num_points', type=int, default=2048)

# Datasets and loaders
parser.add_argument('--train_dataset_choice', type=str, default='mixture',choices=['shapenet', 'ours', 'mixture'])
parser.add_argument('--dataset_path', type=str, default='/data2/tli/normal-auxiliary-diffusion-point-cloud/data/shapenetcore_partanno_segmentation_benchmark_v0_normal')
# parser.add_argument('--syn_dataset_path', type=str, default='./29/logs_interpreter_dir/Generate2_Dataset_bag_1644234931')
# parser.add_argument('--syn_dataset_path', type=str, default='./29/logs_interpreter_dir/Generate2_Dataset_airplane_1644230055')
# parser.add_argument('--syn_dataset_path', type=str, default='./29/logs_interpreter_dir/Generate2_Dataset_chair_1644231130')
parser.add_argument('--syn_dataset_path', type=str, default='./res_partnet_dir/sgm/bag/-1/bag_Sample_10112.npz')
parser.add_argument('--categories', type=str, default='bag')
parser.add_argument('--label_min', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--k_shot', type=int, default=-1)
parser.add_argument('--k_sample', type=int, default=256)
parser.add_argument('--uncertainty_portion', type=float,  default=0.3)
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)

# airplane len dataset,  1958 ,490
# chair len dataset,  2658    ,665
# table len dataset,  3835    ,958
# car len dataset,  659       ,165
# lamp len dataset,  1118     ,280
# guitar len dataset,  550
# bag len dataset,  54

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
parser.add_argument('--log_root', type=str, default='./res_1fig_eva')
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--max_iters', type=int, default=100000)
parser.add_argument('--val_freq', type=int, default=20000)
parser.add_argument('--test_freq', type=int, default=30*THOUSAND)
parser.add_argument('--test_size', type=int, default=400)
parser.add_argument('--tag', type=str, default='_')
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix=args.categories+'_'+args.model, postfix='')
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
logger.info('Building model...')
if args.model == 'pointnet':
    model = PointNetDenseCls(k=args.num_classes, feature_transform=False).to(args.device)
elif args.model == 'pointnetpp':
    model = pointnet2_part_seg_ssg(args.num_classes).to(args.device)
elif args.model == 'dgcnn':
    model = DGCNN_partseg(args, args.num_classes).to(args.device)
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
if args.train_dataset_choice == 'shapenet':
    # TRAIN_DATASET = part_dataset_all_normal.PartNormalDataset(root=args.dataset_path, npoints=2048, split='train', cates=args.categories)
    TRAIN_DATASET = FewShotPartNormalDataset(root=args.dataset_path, npoints=2048, split='train', normalize=False, 
                                             cates=args.categories, k_shot=args.k_shot)
    # trainset_min_label = args.label_min
    dataloader = torch.utils.data.DataLoader( TRAIN_DATASET, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    num_batch = len(TRAIN_DATASET) // args.train_batch_size
    logger.info(('Start training... %s, ........Train shape num: %d') % (args.train_dataset_choice , len(TRAIN_DATASET)))
if args.train_dataset_choice == 'mixture':
    TRAIN_DATASET = FewShotPartNormalDataset(root=args.dataset_path, npoints=2048, split='train', normalize=False, 
                                             cates=args.categories, k_shot=args.k_shot)
    # trainset_min_label = args.label_min
    img, cl, target = TRAIN_DATASET.__getitem__(3)
    print('TRAIN SET',img.shape, cl.shape, target.shape)

    # arr = np.load(os.path.join(args.syn_dataset_path,'samples_2048x3.npz')).values()
    arr = np.load(args.syn_dataset_path).values()
    if len(arr) == 3:
        pcs, labels, uncertainty_scores = arr
    else: # Needed to handle datasetGAN
        pcs, labels, latents, uncertainty_scores = arr
    # all_pcs = pcs[::2]
    # all_labels = labels[::2]
    # all_uncertainty_scores = uncertainty_scores[::2]
    # idxs = np.argsort(all_uncertainty_scores)
    # pcs = all_pcs[idxs[30:(args.k_sample+30)]]
    # labels = all_labels[idxs[30:(args.k_sample+30)]]

    all_pcs = pcs[:2000,:]
    all_labels = labels[:2000,:]
    all_uncertainty_scores = uncertainty_scores[:2000]
    idxs = np.argsort(all_uncertainty_scores)
    filter_out_num = int(len(idxs) * args.uncertainty_portion)
    idxs = idxs[30: -filter_out_num + 30]
    print('all_pcs = ', all_pcs.shape, all_labels.shape, all_uncertainty_scores.shape, filter_out_num, idxs.shape)
    pcs = all_pcs[idxs]
    labels = all_labels[idxs]

    # print('all_pcs = ', pcs.shape, uncertainty_scores.shape, all_uncertainty_scores.shape, idxs.shape)
    for i in range(pcs.shape[0]):
        pc = pcs[i]
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        pcs[i] = pc
    TRAIN_DATASET1 = InMemoryPCLabelDataset(
        pcs=pcs,
        labels=labels
    )
    img, cl, target = TRAIN_DATASET1.__getitem__(3)
    print('TRAIN SET',img.shape, cl.shape, target.shape)
    CONCAT_DATASET = ConcatDataset([TRAIN_DATASET,TRAIN_DATASET1])
    dataloader = torch.utils.data.DataLoader( CONCAT_DATASET, batch_size=args.train_batch_size, shuffle=True, drop_last=True, num_workers=int(args.num_workers))
    num_batch = len(CONCAT_DATASET) // args.train_batch_size
    logger.info(('Start training... %s, ........Train shape num: %d') % (args.train_dataset_choice , len(CONCAT_DATASET)))

# print(args.train_dataset_choice, len(TRAIN_DATASET1))
# for i in range(10):
#     pc, _, label = TRAIN_DATASET1[i+300]
#     color = np.zeros((2048,3))
#     labels = label
#     for j in range(2048):
#         if labels[j] == 0.0:
#             color[j] = np.array([0.7,0.7,0.3])
#         if labels[j] == 1.0:
#             color[j] = np.array([0.3,0.3,0.7])
#         if labels[j] == 2.0:
#             color[j] = np.array([0.3,0.7,0.3])
#         if labels[j] == 3.0:
#             color[j] = np.array([0.7,0.3,0.3])
#     pcu.save_mesh_vc(os.path.join(args.log_root,f'y{i}.ply'), pc, color)

i = 0
for epoch in range(20):
    for data in enumerate(dataloader):
        points, _, target = data[1]
        
        points = points.to(dtype=torch.float32)
        target = (target).to(dtype=torch.long)
        print('TRAIN target:  ', target.min(), target.max())


        points = points.transpose(2, 1)
        points, target = points.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        model = model.train()
        if args.model == 'pointnet':
            pred, _, trans_feat = model(points)
        elif args.model == 'pointnetpp':
            label = torch.zeros((points.shape[0],1)).to(device=points.device, dtype=points.dtype)
            pred, _, trans_feat = model(points, to_categorical(label, 16))
        elif args.model == 'dgcnn':
            label = torch.zeros((points.shape[0])).to(device=points.device, dtype=points.dtype)
            pred, _, trans_feat = model(points, to_categorical(label, 16))
        pred = pred.contiguous().view(-1, args.num_classes)
        target = target.view(-1, 1)[:, 0]
        print('here ', points.shape, pred.size(), target.size())
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
testdataloader1 = DataLoader(TEST_DATASET1, batch_size=4, shuffle=True, drop_last=True)
shape_ious = []
logger.info('Start evaluating of ShapeNet....')
for i,data in tqdm(enumerate(testdataloader1, 0)):
    points, _, target = data
    # pcu.save_mesh_v(os.path.join(log_dir,'val.ply'), points[0].detach().cpu().numpy())
    points = points.to(dtype=torch.float32)
    target = (target).to(dtype=torch.long)
    print('TEST target:  ', target.min(), target.max())

    points = points.transpose(2, 1)
    points, target = points.to(args.device), target.to(args.device)
    # model = model.eval()
    # pred, _, _ = model(points)
    if args.model == 'pointnet':
        pred, _, _ = model(points)
    elif args.model == 'pointnetpp':
        label = torch.zeros((points.shape[0],1)).to(device=points.device, dtype=points.dtype)
        pred, _, _ = model(points, to_categorical(label, 16))
    elif args.model == 'dgcnn':
        label = torch.zeros((points.shape[0])).to(device=points.device, dtype=points.dtype)
        pred, _, trans_feat = model(points, to_categorical(label, 16))
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