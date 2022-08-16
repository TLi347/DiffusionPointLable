import imp
import sys
import torch
from torch import nn
from typing import List

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.common import *
from models.autoencoder import *

from collections import OrderedDict
from compare.UnsupervisedPointCloudReconstruction.modelbk import ReconstructionNet
from compare.PointFlow.models.networksbk import PointFlow
from compare.caps.models.pointcapsnet_ae import PointCapsNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_feature_extractor(args):
    """ Create the feature extractor for <model_type> architecture. """
    if args.model == 'fold':
        print("Creating Fold Feature Extractor...")
        feature_extractor = FeatureExtractorFold(args)
    elif args.model == 'flow':
        print("Creating Flow Feature Extractor...")
        feature_extractor = FeatureExtractorFlow(args)
    elif args.model == 'caps':
        print("Creating Caps Feature Extractor...")
        feature_extractor = FeatureExtractorCaps(args)
    elif args.model == 'sgm':
        print("Creating SGM Feature Extractor...")
        feature_extractor = FeatureExtractorSGM(args)
    elif args.model == 'sgmae':
        print("Creating SGM Feature Extractor...")
        feature_extractor = FeatureExtractorSGMAE(args)
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
    save_tensors(self, out, 'output')
    return out


def save_input_hook(self, inp, out):
    save_tensors(self, inp[0], 'input')
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

class FeatureExtractorFold(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        layers = [self.model.decoder.folding1, self.model.decoder.folding2]

        for layer in self.model.decoder.folding1:
            if type(layer) == nn.Conv1d:
                if layer.out_channels != 3:
                    layer.register_forward_hook(self.save_hook)
                    self.feature_blocks.append(layer)
                    break
        for layer in self.model.decoder.folding2:
            if type(layer) == nn.Conv1d:
                if layer.out_channels != 3:
                    layer.register_forward_hook(self.save_hook)
                    self.feature_blocks.append(layer)
                    break


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
        self.model(x)
        # output, _ = self.model(x)
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_partnet_dir/tmp/t3-fold-in-chair.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_partnet_dir/tmp/t3-fold-out-chair.ply',output[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.output)
            block.output = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorFlow(FeatureExtractor):
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
        
        # layer = self.model.point_cnf.chain[0]
        # layer.register_forward_hook(self.save_hook)
        # self.feature_blocks.append(layer)
        
        # layer = self.model.point_cnf.chain[1].odefunc.diffeq.layers[0]._layer
        # layer.register_forward_hook(self.save_hook)
        # self.feature_blocks.append(layer)
        # layer = self.model.point_cnf.chain[1].odefunc.diffeq.layers[1]._layer
        # layer.register_forward_hook(self.save_hook)
        # self.feature_blocks.append(layer)
        layer = self.model.point_cnf.chain[1].odefunc.diffeq.layers[2]._layer
        layer.register_forward_hook(self.save_hook)
        self.feature_blocks.append(layer)
    
    def _load_pretrained_model(self, args):

        self.model = PointFlow(args)
        self.model = self.model#.cuda()

        state_dict = torch.load(args.flow_ckpt,map_location='cpu')
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if 'module' in key:
                name = key.replace('module.','')
            else:
                name = key
            new_state_dict[name] = val
        # print(self.model.encoder.conv1.weight.shape)
        # print(new_state_dict['encoder.conv1.weight'].shape)
        self.model.load_state_dict(new_state_dict)
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        self.model.reconstruct(x,num_points=2048)
        # output = self.model.reconstruct(x,num_points=2048)
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_tab3_dir/flow-in-chair.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_tab3_dir/flow-out-chair.ply',output[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.output)
            block.output = None

        # Per-layer list of activations [N, C, H, W]
        return activations


class FeatureExtractorCaps(FeatureExtractor):
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
            layer.conv2.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer.conv2)
            layer.conv3.register_forward_hook(self.save_hook)
            self.feature_blocks.append(layer.conv3)

    def _load_pretrained_model(self, args):
        self.model = PointCapsNet(args.caps_prim_caps_size, args.caps_prim_vec_size, args.caps_latent_caps_size, args.caps_latent_caps_size, args.sample_num_points)
        self.model.load_state_dict(torch.load(args.caps_ckpt,map_location='cpu'))
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        activations = []
        self.model(x.transpose(2, 1).contiguous())
        # _, output = self.model(x.transpose(2, 1).contiguous())
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('./res_tab3_dir/caps-in-chair.ply',x[0].detach().cpu().numpy())
        # pcu.save_mesh_v('./res_tab3_dir/caps-out-chair.ply',output.transpose(2, 1)[0].detach().cpu().numpy())

        # Extract activations
        for block in self.feature_blocks:
            activations.append(block.output)
            block.output = None

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
        
        # # Save decoder activations
        # for idx, block in enumerate(self.model.diffusion.net.layers):
        #     if idx in args.sgm_blocks:
        #         block._layer.register_forward_hook(self.save_hook)
        #         self.feature_blocks.append(block._layer)
        # Save decoder activations
        for idx, block in enumerate(self.model.diffusion.net.layers):
            if idx in args.sgm_blocks:
                block._layer.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._layer)
                
                block._hyper_gate.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._hyper_gate)
                
                block._hyper_bias.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._hyper_bias)

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
        # w = torch.randn([1, 256]).to(x.device)
        # context = self.model.flow(w, reverse=True).view(1, -1)
        # batch_size = context.size(0)
        # x_T = torch.randn([batch_size, 2048, 3]).to(context.device)
        # traj = {self.model.diffusion.var_sched.num_steps: x_T}
        # for t in range(self.model.diffusion.var_sched.num_steps, 0, -1):
        #     z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
        #     alpha = self.model.diffusion.var_sched.alphas[t]
        #     alpha_bar = self.model.diffusion.var_sched.alpha_bars[t]
        #     sigma = self.model.diffusion.var_sched.get_sigmas(t, 0.0)

        #     c0 = 1.0 / torch.sqrt(alpha)
        #     c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

        #     x_t = traj[t]
        #     beta = self.model.diffusion.var_sched.betas[[t]*batch_size]
        #     e_theta = self.model.diffusion.net(x_t, beta=beta, context=context)
        #     x_next = c0 * (x_t - c1 * e_theta) + sigma * z
        #     traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
        #     traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('/data2/tli/seg-diffusion-point-cloud/res_partnet_dir/tmp/bag-traj.ply',traj[0][0].detach().cpu().numpy())

        activations = []
        z_mu, z_sigma = self.model.encoder(x)
        context = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.ones(x.size(0)).to(device=x.device,dtype=torch.int64) * t
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

            # # Extract activations
            # for block in self.feature_blocks:
            #     activations.append(block.output)
            #     block.output = None
            block_x = self.feature_blocks[0].output
            self.feature_blocks[0].output = None
            block_gate = self.feature_blocks[1].output
            self.feature_blocks[1].output = None
            block_bias = self.feature_blocks[2].output
            self.feature_blocks[2].output = None
            activations.append( block_x * block_gate + block_bias )

        # Per-layer list of activations [N, C, H, W]
        return activations
    
    @torch.no_grad()
    def sample(self, x, noise=None):
        batch_size = x.size(0)
        w = torch.randn([batch_size, 256]).to(x.device)
        context = self.model.flow(w, reverse=True).view(batch_size, -1)
        x_T = torch.randn([batch_size, 2048, 3]).to(context.device)
        traj = {self.model.diffusion.var_sched.num_steps: x_T}
        
        activations = []
        for t in range(self.model.diffusion.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.model.diffusion.var_sched.alphas[t]
            alpha_bar = self.model.diffusion.var_sched.alpha_bars[t]
            sigma = self.model.diffusion.var_sched.get_sigmas(t, 0.0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.model.diffusion.var_sched.betas[[t]*batch_size]
            
            if t in self.steps:
                self.model.diffusion.net(x_t, beta=beta, context=context)
                block_x = self.feature_blocks[0].output
                self.feature_blocks[0].output = None
                block_gate = self.feature_blocks[1].output
                self.feature_blocks[1].output = None
                block_bias = self.feature_blocks[2].output
                self.feature_blocks[2].output = None
                activations.append( block_x * block_gate + block_bias )
            
            e_theta = self.model.diffusion.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
        # import point_cloud_utils as pcu
        # pcu.save_mesh_v('/data2/tli/seg-diffusion-point-cloud/res_partnet_dir/tmp/table-traj.ply',traj[0][0].detach().cpu().numpy())
        # Per-layer list of activations [N, C, H, W]
        return activations, traj[0]


class FeatureExtractorSGMAE(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, args):
        super().__init__(args)
        self.steps = args.sgmae_steps
        
        # Save decoder activations
        for idx, block in enumerate(self.model.diffusion.net.layers):
            if idx in args.sgmae_blocks:
                block._layer.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._layer)

    def _load_pretrained_model(self, args):
        
        ckpt = torch.load(args.sgmae_ckpt,map_location='cpu')
        self.flexibility = ckpt['args'].flexibility
        self.model = AutoEncoder(ckpt['args'])
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(args.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):
        code = self.model.encode(x)
        output = self.model.decode(code, x.size(1), flexibility=self.flexibility).detach()
        import point_cloud_utils as pcu
        pcu.save_mesh_v('./res_tab3_dir/t3-sgmae-in-chair.ply',x[0].detach().cpu().numpy())
        pcu.save_mesh_v('./res_tab3_dir/t3-sgmae-out-chair.ply',output[0].detach().cpu().numpy())

        activations = []
        context, _ = self.model.encoder(x)
        for t in self.steps:
            # Compute x_t and run DDPM
            t = torch.ones(x.size(0)).to(device=x.device,dtype=torch.int64) * t
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
                activations.append(block.output)
                block.output = None

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
            feats, size=size, mode=args.upsample_mode, align_corners=False
        )
        resized_activations.append(feats[0])
    
    # for i in range(len(resized_activations)):
    #     print(i, resized_activations[i].shape)
    # print('1 resized_activations, ', torch.cat(resized_activations, dim=-1).transpose(1,0).shape)
    if args.model == 'caps':
        return torch.cat(resized_activations, dim=-1).transpose(1,0)
    return torch.cat(resized_activations, dim=-1)


def collect_features_batch(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = args['features_dim'][-1]
    resized_activations = []
    for feats in activations:
        if args['model'] == 'fold':
            feats = feats.transpose(1, 2)
        if args['model'] == 'caps':
            feats = feats.transpose(1, 2)
        feats = nn.functional.interpolate(
            feats, size=size, mode=args['upsample_mode']
        )
        resized_activations.append(feats)
    if args['model'] == 'caps':
        return torch.cat(resized_activations, dim=1)
    # import point_cloud_utils as pcu
    # y = torch.cat(resized_activations, dim=-1)[3]
    # pcu.save_mesh_v('./res_tab3_dir/t3-fold-in-y.ply', y.detach().cpu().numpy())
    return torch.cat(resized_activations, dim=-1)