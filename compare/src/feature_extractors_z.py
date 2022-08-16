import imp
import sys
import torch
from torch import nn
from typing import List

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.common import *

import point_cloud_utils as pcu
from collections import OrderedDict
from compare.UnsupervisedPointCloudReconstruction.modelbk import ReconstructionNet
from compare.PointFlow.models.networksbk import PointFlow
from compare.caps.models.pointcapsnet_ae import PointCapsNet
from compare.spgan.Generation.model_test import Model
from torch.autograd import Variable


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_feature_extractor_z(args):
    """ Create the feature extractor for <model_type> architecture. """
    if args.model == 'spgan':
        print("Creating SPGAN Feature Extractor...")
        feature_extractor = FeatureExtractorSPGAN(args)
    elif args.model == 'sgm':
        print("Creating SGM Feature Extractor...")
        feature_extractor = FeatureExtractorSGM(args)
    elif args.model == 'sgmz':
        print("Creating SGMz Feature Extractor...")
        feature_extractor = FeatureExtractorSGM_z(args)
    elif args.model == 'fold':
        print("Creating Fold Feature Extractor...")
        feature_extractor = FeatureExtractorFold_z(args)
    elif args.model == 'flow':
        print("Creating Flow Feature Extractor...")
        feature_extractor = FeatureExtractorFlow_z(args)
    elif args.model == 'caps':
        print("Creating Caps Feature Extractor...")
        feature_extractor = FeatureExtractorCaps_z(args)
    else:
        raise Exception(f"Wrong model type: {args.model}")
    return feature_extractor

def save_tensors(module: nn.Module, features, name: str):
    """ Process and save activations in the module. """
    if type(features) in [list, tuple]:
        features = [f.detach().float() if f is not None else None 
                    for f in features]
        setattr(module, name, features)
    elif isinstance(features, dict):
        features = {k: f.detach().float() for k, f in features.items()}
        setattr(module, name, features)
    else:
        setattr(module, name, features.detach().float())


def save_out_hook(self, inp, out):
    save_tensors(self, out, 'activations')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'activations')
    return out

class FeatureExtractor(nn.Module):
    # def __init__(self, model_path: str, input_activations: bool, **kwargs):
    def __init__(self, args):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        # self._load_pretrained_model(model_path, **kwargs)
        self._load_pretrained_model(args)
        print(f"Pretrained model is successfully loaded from ")
        self.save_hook = save_input_hook if args.input_activations else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, args):
        pass



class FeatureExtractorSGM_z(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        self.steps = args.sgm_steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.diffusion.net.layers):
            if idx in args.sgm_blocks:
                block._layer.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._layer)

    def _load_pretrained_model(self, args):
        
        ckpt = torch.load(args.sgm_ckpt,map_location='cpu')
        if ckpt['args'].model == 'gaussian':
            self.model = GaussianVAE(ckpt['args'])
        elif ckpt['args'].model == 'flow':
            self.model = FlowVAE(ckpt['args'])
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        z_mu, z_sigma = self.model.encoder(x)
        context = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            # noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            # self.model(noisy_x, self.diffusion._scale_timesteps(t))
            alpha_bar = self.model.diffusion.var_sched.alpha_bars[t]
            beta = self.model.diffusion.var_sched.betas[t]

            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

            if noise is None:
                e_rand = torch.randn_like(x)  # (B, N, d)
            else:
                e_rand = noise
            self.model.diffusion.net(c0 * x + c1 * e_rand, beta=beta, context=context)

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations



class FeatureExtractorSGM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        self.steps = args.sgm_steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.diffusion.net.layers):
            if idx in args.sgm_blocks:
                block._layer.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._layer)

    def _load_pretrained_model(self, args):
        
        ckpt = torch.load(args.sgm_ckpt,map_location='cpu')
        if ckpt['args'].model == 'gaussian':
            self.model = GaussianVAE(ckpt['args'])
        elif ckpt['args'].model == 'flow':
            self.model = FlowVAE(ckpt['args'])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(args.device)
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        z_mu, z_sigma = self.model.encoder(x)
        context = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.tensor([t]).to(x.device)
            # noisy_x = self.diffusion.q_sample(x, t, noise=noise)
            # self.model(noisy_x, self.diffusion._scale_timesteps(t))
            alpha_bar = self.model.diffusion.var_sched.alpha_bars[t]
            beta = self.model.diffusion.var_sched.betas[t]

            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

            if noise is None:
                e_rand = torch.randn_like(x)  # (B, N, d)
            else:
                e_rand = noise
            self.model.diffusion.net(c0 * x + c1 * e_rand, beta=beta, context=context)
            # pcu.save_mesh_v(f'./results/time{t.cpu().numpy()}.ply', (c0 * x + c1 * e_rand)[0].detach().cpu().numpy())

            # Extract activations
            for block in self.feature_blocks:
                activations.append(block.activations)
                block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorSPGAN(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        
        # self.model.adain1.style.register_forward_hook(self.save_hook)
        self.model.adain1.norm.register_forward_hook(self.save_hook)
        self.feature_blocks.append(self.model.adain1.norm)
        # self.model.adain2.style.register_forward_hook(self.save_hook)
        self.model.adain2.norm.register_forward_hook(self.save_hook)
        self.feature_blocks.append(self.model.adain2.norm)

    def _load_pretrained_model(self, args):
        
        args.pretrain_model_G = "airplane_G.pth"
        args.log_dir = "compare/SP-GAN/models"
        self.model = Model(args)
        self.model.build_model_eval()
        could_load, save_epoch = self.model.load(self.opts.log_dir)
        self.model.G.eval()
        self.model.G.to(args.device)

    @torch.no_grad()
    def forward(self, x):
        activations = []
        ball = self.model.read_ball()
        x = np.expand_dims(ball, axis=0)
        number = x.shape[0]
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x)).cuda()
        noise = np.random.normal(0, 0.2, (number, self.opts.nz))
        noise = np.expand_dims(noise,axis=1)
        noise = np.tile(noise, (1, self.opts.np, 1))# [B,128]->[B,1,128]->[B,2048,128]
        z = Variable(torch.Tensor(noise)).cuda()
        output = self.G(x, z)
        output = output.transpose(2,1)
        import point_cloud_utils as pcu
        pcu.save_mesh_v('./res_tab4_dir/t3-spgan-in.ply',x[0].detach().cpu().numpy())
        pcu.save_mesh_v('./res_tab4_dir/t3-spgan-out.ply',output[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations





class FeatureExtractorFold_z(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        layers = [self.model.decoder.folding1, self.model.decoder.folding2]

        self.model.decoder.folding2[2].register_forward_hook(self.save_hook)
        self.feature_blocks.append(self.model.decoder.folding2[2])
        # for layer in self.model.decoder.folding1:
        #     if type(layer) == nn.Conv1d:
        #         if layer.out_channels != 3:
        #             layer.register_forward_hook(self.save_hook)
        #             self.feature_blocks.append(layer)
        #     break

    def _load_pretrained_model(self, args):
        
        self.model = ReconstructionNet(args)
        state_dict = torch.load(args.folding_ckpt, map_location='cpu')
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        pc_max, _ = x[0].max(dim=0, keepdim=True) # (1, 3)
        pc_min, _ = x[0].min(dim=0, keepdim=True) # (1, 3)
        shift = ((pc_min + pc_max) / 2).view(1, 3)
        scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        x = (x - shift) / scale
        output, _ = self.model(x)
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_tab3_dir/t31-fold-in.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_tab3_dir/t31-fold-out.ply',output[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorFlow_z(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        layers = [self.model.point_cnf.chain[1].odefunc.diffeq.layers[1]._layer, 
                  self.model.point_cnf.chain[1].odefunc.diffeq.layers[2]._layer]

        # layer = self.model.point_cnf.chain[1].odefunc.diffeq.layers[1]._layer
        # layer.register_forward_hook(self.save_hook)
        # self.feature_blocks.append(layer)
        layer = self.model.point_cnf.chain[1].odefunc.diffeq.layers[3]._layer
        layer.register_forward_hook(self.save_hook)
        self.feature_blocks.append(layer)
    
    def _load_pretrained_model(self, args):

        self.model = PointFlow(args)
        self.model = self.model.cuda()

        state_dict = torch.load(args.flow_ckpt)
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if 'module' in key:
                name = key.replace('module.','')
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        output = self.model.reconstruct(x,num_points=2048)
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_tab3_dir/t3-flow-in.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_tab3_dir/t3-flow-out.ply',output[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorCaps_z(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        layers = self.model.caps_decoder.decoder

        for layer in layers:
            layer.conv1.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer.conv1)

    def _load_pretrained_model(self, args):
        self.model = PointCapsNet(args.caps_prim_caps_size, args.caps_prim_vec_size, args.caps_latent_caps_size, args.caps_latent_caps_size, args.sample_num_points)
        self.model.load_state_dict(torch.load(args.caps_ckpt))
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        _, output = self.model(x.transpose(2, 1).contiguous())
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_tab3_dir/t3_caps_in.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_tab3_dir/t3_caps_out.ply',output.transpose(2, 1)[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.activations)
            block.activations = None

        # Per-layer list of activations [N, C, H, W]
        return activations





def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = args.features_dim[-1]
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        if args.model == 'fold':
            feats = feats.transpose(1, 2)
        feats = nn.functional.interpolate(
            feats, size=size, mode=args.upsample_mode, align_corners=True
        )
        resized_activations.append(feats[0])
    
    # for i in range(len(resized_activations)):
    #     print(i, resized_activations[i].shape)
    # print('1 resized_activations, ', torch.cat(resized_activations, dim=-1).transpose(1,0).shape)
    if args.model == 'caps':
        return torch.cat(resized_activations, dim=-1).transpose(1,0)
    return torch.cat(resized_activations, dim=-1)