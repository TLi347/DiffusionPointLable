import sys
import torch
from torch import nn
from typing import List

from utils.misc import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.common import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """
    if model_type == 'ddpm':
        print("Creating DDPM Feature Extractor...")
        feature_extractor = FeatureExtractorDDPM(**kwargs)
    else:
        raise Exception(f"Wrong model type: {model_type}")
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
    def __init__(self, **kwargs):
        ''' 
        Parent feature extractor class.
        
        param: model_path: path to the pretrained model
        param: input_activations: 
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        # self._load_pretrained_model(model_path, **kwargs)
        self._load_pretrained_model(**kwargs)
        print(f"Pretrained model is successfully loaded from {kwargs['ckpt']}")
        self.save_hook = save_input_hook if kwargs['input_activations'] else save_out_hook
        self.feature_blocks = []

    def _load_pretrained_model(self, **kwargs):
        pass

class FeatureExtractorDDPM(FeatureExtractor):
    ''' 
    Wrapper to extract features from pretrained DDPMs.
            
    :param steps: list of diffusion steps t.
    :param blocks: list of the UNet decoder blocks.
    '''
    
    # def __init__(self, steps: List[int], blocks: List[int], **kwargs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.steps = kwargs['steps']
        
        # Save decoder activations
        for idx, block in enumerate(self.model.diffusion.net.layers):
            if idx in kwargs['blocks']:
                block._layer.register_forward_hook(self.save_hook)
                self.feature_blocks.append(block._layer)

    def _load_pretrained_model(self, **kwargs):
        
        ckpt = torch.load(kwargs['ckpt'])
        seed_all(kwargs['seed'])
        if ckpt['args'].model == 'gaussian':
            self.model = GaussianVAE(ckpt['args'])
        elif ckpt['args'].model == 'flow':
            self.model = FlowVAE(ckpt['args'])
        self.model.to(kwargs['device'])
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

def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = args['features_dim'][-1]
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])
    
    # for i in range(len(resized_activations)):
    #     print(i, resized_activations[i].shape)
    # print('1 resized_activations, ', torch.cat(resized_activations, dim=-1).shape)
    return torch.cat(resized_activations, dim=-1)