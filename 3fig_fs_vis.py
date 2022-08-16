
import os
os.chdir('/data2/tli/seg-diffusion-point-cloud')
import math
import argparse
from xmlrpc.client import boolean
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
import torch.utils.tensorboard
from torch.utils.data import DataLoader, dataset
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import point_cloud_utils as pcu
import time

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.part_dataset_all_normal import PartNormalDataset
from utils.few_shot_part_dataset_all_normal import FewShotPartNormalDataset
from utils.partnet_dataset import PartNetDataset
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from compare.UnsupervisedPointCloudReconstruction.modelbk import ReconstructionNet
from collections import OrderedDict
from compare.PointFlow.models.networksbk import PointFlow
from compare.caps.models.pointcapsnet_ae import PointCapsNet

from models.flow import add_spectral_norm, spectral_norm_power_iteration
os.chdir('/data2/tli/seg-diffusion-point-cloud')

# Arguments
parser = argparse.ArgumentParser()
#>> Interperter arguments
parser.add_argument('--model', type=str, default='sgm', choices=['fold', 'flow', 'caps', 'sgm', 'sgmae'])
parser.add_argument('--share_noise', type=bool, default=True)

#>> Interperter Datasets and loaders
parser.add_argument('--categories', type=str, default='bag')
parser.add_argument('--label_min', type=int, default=4)
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--label_data_num', type=int, default=3100) # 2658
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--k_shot', type=int, default=-1) # 2658

# SGM
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_01_23__15_02_13/dpm-baseline-car-benchmark_v0x3-10w.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_02_18__01_13_27/ckpt_0.000000_240000.pt')#airplane
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_01_17__20_21_43/dpm-baseline-airplane-benchmark_v0x3-10w.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_06_21__22_49_44/ckpt_0.000000_300000.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./29/logs_gen/GEN_2021_12_28__20_56_55/dpm-baseline-airplane-benchmark_v0-10w.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./29/logs_gen/GEN_2021_12_24__13_15_09/dpm-baseline-chair-becchmark_v0-10w.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_05_22__20_42_21/ckpt_0.000000_102000.pt') #chair
# parser.add_argument('--sgm_ckpt', type=str, default='/data2/tli/seg-diffusion-point-cloud/logs_gen/GEN_2022_06_21__22_49_44/ckpt_0.000000_300000.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_02_25__21_33_56/ckpt_dpm_table.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_02_26__15_25_45/ckpt_dpm_lamp.pt')
# parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_02_26__15_24_16/ckpt_dpm_car.pt')
parser.add_argument('--sgm_ckpt', type=str, default='./logs_gen/GEN_2022_08_14__20_30_39/ckpt_0.000000_40000.pt') #bag
parser.add_argument('--sgm_steps', type=list, default=[1,5,10,15,20])
parser.add_argument('--sgm_blocks', type=list, default=[0])
parser.add_argument('--sgm_upsample_mode', type=str, default='linear')
# SGMAE
parser.add_argument('--sgmae_ckpt', type=str, default='./logs_ae/AE_2022_06_22__14_50_50/ckpt_0.020244_324000.pt')# chair
# parser.add_argument('--sgmae_ckpt', type=str, default='./logs_ae/AE_2022_06_20__11_55_24/ckpt_0.008812_81000.pt')
parser.add_argument('--sgmae_steps', type=list, default=[1,20,40,60])
parser.add_argument('--sgmae_blocks', type=list, default=[0])
parser.add_argument('--sgmae_upsample_mode', type=str, default='linear')
# Fold
parser.add_argument('--folding_ckpt', type=str, default='./compare/UnsupervisedPointCloudReconstruction/snapshot/Reconstruct_foldnet_gaussian/models/shapenetcorev2_278.pkl')
# parser.add_argument('--folding_ckpt', type=str, default='./compare/UnsupervisedPointCloudReconstruction/snapshot/Reconstruct_chair/models/shapenetcorev2_best.pkl')
parser.add_argument('--folding_encoder', type=str, default='foldnet', metavar='N', choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'])
parser.add_argument('--folding_feat_dims', type=int, default=512, metavar='N')
parser.add_argument('--folding_dropout', type=int, default=0.5, metavar='N')
parser.add_argument('--folding_k', type=int, default=16, metavar='N')
parser.add_argument('--folding_shape', type=str, default='gaussian', metavar='N',choices=['plane', 'sphere', 'gaussian'])
# Flow
parser.add_argument('--flow_ckpt', type=str, default='./compare/PointFlow/pretrained_models/ae/airplane/checkpoint.pt')
# parser.add_argument('--flow_ckpt', type=str, default='./compare/PointFlow/pretrained_models/ae/chair/checkpoint.pt')
# parser.add_argument('--flow_ckpt', type=str, default='./compare/PointFlow/pretrained_models/ae/car/checkpoint.pt')
parser.add_argument('--flow_use_latent_flow', action='store_true', help='Whether to use the latent flow to model the prior.')
parser.add_argument('--flow_input_dim', type=int, default=3)
parser.add_argument('--flow_latent_dims', type=str, default='256')
parser.add_argument('--flow_num_blocks', type=int, default=1)
parser.add_argument("--flow_latent_num_blocks", type=int, default=1)
parser.add_argument('--flow_zdim', type=int, default=128)
parser.add_argument('--flow_recon_weight', type=float, default=1.)
parser.add_argument('--flow_prior_weight', type=float, default=1)
parser.add_argument('--flow_entropy_weight', type=float, default=1.)
NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]
parser.add_argument('--flow_layer_type', type=str, default="concatsquash", choices=LAYERS)
parser.add_argument('--flow_time_length', type=float, default=0.5)
parser.add_argument('--flow_train_T', type=eval, default=True, choices=[True, False])
parser.add_argument("--flow_nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
parser.add_argument('--flow_use_adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--flow_solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--flow_atol', type=float, default=1e-5)
parser.add_argument('--flow_rtol', type=float, default=1e-5)
parser.add_argument('--flow_batch_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--flow_sync_bn', type=eval, default=False, choices=[True, False])
parser.add_argument('--flow_bn_lag', type=float, default=0)
parser.add_argument('--flow_distributed', action='store_true')
parser.add_argument('--flow_dims', type=str, default='512-512-512')
parser.add_argument('--flow_use_deterministic_encoder', type=bool, default=True)
parser.add_argument('--flow_evaluate_recon', type=bool, default=True)
# Caps
# opt.prim_caps_size, opt.prim_vec_size, opt.latent_caps_size, opt.latent_caps_size, opt.num_points
parser.add_argument('--caps_ckpt', type=str, default='./compare/3D-point-capsule-networks/checkpoints/shapenet_part_dataset_ae_200.pth')
parser.add_argument('--caps_prim_caps_size', type=int, default=1024, help='number of primary point caps')
parser.add_argument('--caps_prim_vec_size', type=int, default=16, help='scale of primary point caps')
parser.add_argument('--caps_latent_caps_size', type=int, default=64, help='number of latent caps')
parser.add_argument('--caps_latent_vec_size', type=int, default=64, help='scale of latent caps')
# parser.add_argument('--caps_num_points', type=int, default=2048, help='input point set size')

#>> Interperter arguments
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--input_activations', type=bool, default=False)
parser.add_argument('--start_model_num', type=int, default=0)
parser.add_argument('--model_num', type=int, default=10)
parser.add_argument('--model_total', type=int, default=10)
parser.add_argument('--upsample_mode', type=str, default='linear')

# mIoU
# Evaluation Dataset arguments
parser.add_argument('--train_dataset_choice', type=str, default='shapenet',choices=['shapenet', 'ours', 'modelnet40', 'modelnet10'])
parser.add_argument('--dataset_path', type=str, default='/data2/tli/normal-auxiliary-diffusion-point-cloud/data/shapenetcore_partanno_segmentation_benchmark_v0_normal')
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--num_workers', type=int, default=4)

# Training
parser.add_argument('--total_epcho_num', type=int, default=800)
parser.add_argument('--decay_freq', type=int, default=200)
parser.add_argument('--init_lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=9988)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--save_dir', type=str, default='./res_partnet_dir')
parser.add_argument('--device', type=str, default='cuda:1')
args = parser.parse_args()
seed_all(args.seed)

if args.model == 'fold':
    # Fold
    args.features_dim = [1024]
    args.dim = [2048,1024]
    args.dataset_normalize = True
    pc_scale = 1
if args.model == 'flow':
    # Flow
    args.features_dim = [1024]
    args.dim = [2048,1024]
    args.dataset_normalize = False
    pc_scale = 1
if args.model == 'caps':
    # Caps
    args.features_dim = [1024]
    args.dim = [64*32,1024]
    args.dataset_normalize = True
    pc_scale = 1
if args.model == 'sgm':
    # SGM
    args.features_dim = [256]
    args.dim = [2048,256*5]
    args.dataset_normalize = False
    pc_scale = 1
if args.model == 'sgmae':
    # SGM
    args.features_dim = [512]
    args.dim = [2048,512*4]
    args.dataset_normalize = False
    pc_scale = 1

# # Logging
# save_dir = os.path.join(args.save_dir, '%s' % (args.model), '49_44_%s' % (args.categories), '%s' % (str(args.k_shot)) )
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# logger = get_logger('3fig_fs_train_', save_dir)
# for k, v in vars(args).items():
#     logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
seed_all(args.seed)

from compare.src.feature_extractors import create_feature_extractor, collect_features, collect_features_batch
from src.datasets import FeatureDataset
from compare.src.pixel_classifier import pixel_classifier, load_ensemble, predict_labels
from src.utils import multi_acc
import json

args.start_model_num = 0

# Datasets and loaders
feature_extractor = create_feature_extractor(args)
opts = json.load(open('./logs_exp_dir/ddpm.json', 'r'))
opts.update(vars(args))
print('Loading datasets...')
dataset = FewShotPartNormalDataset(root=args.dataset_path, split='train', normalize=args.dataset_normalize, npoints=2048, cates=args.categories, k_shot=args.k_shot)
dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
print('len dataset, ', len(dataset))

# Classifier training
print('Start Classfier train...')
if 'share_noise' in args:
    rnd_gen = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.randn(1, 2048, 3, generator=rnd_gen, device=args.device).repeat(args.batch_size,1,1)
else:
    noise = None

for data in dataloader:
    # print(f"Preparing the train set for {opts['categories']}...")
    pc = (data[0]).to(device=opts['device'],dtype=torch.float32)#.repeat(args.batch_size,1,1)
    features = feature_extractor(pc, noise=noise)
    X = collect_features_batch(opts, features).cpu()

    for i in range(args.batch_size):
        features = X[i].cpu().detach().numpy()
        kmeans = KMeans(n_clusters=args.num_class)
        kmeans.fit(features)
        y_kmeans = kmeans.predict(features)
        print('@@@@@@@')
        from datetime import datetime
        from src.utils import *
        color = np.ones((2048,4))
        for j in range(2048):
            if y_kmeans[j] == 0.0:
                color[j] = np.array([0.7,0.7,0.3,1])
            if y_kmeans[j] == 1.0:
                color[j] = np.array([0.3,0.3,0.7,1])
            if y_kmeans[j] == 2.0:
                color[j] = np.array([0.3,0.7,0.3,1])
            if y_kmeans[j] == 3.0:
                color[j] = np.array([0.7,0.3,0.3,1])
        pcu.save_mesh_vc(os.path.join('./res_partnet_dir/tmp', args.model+f'{i}.ply'), pc[i].cpu().detach().numpy(), color)
    break

