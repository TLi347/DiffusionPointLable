import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Conv1d(dim, 128,1),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Conv1d(128, 32, 1),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Conv1d(32, numpy_class, 1)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv1d(dim, 256, 1),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Conv1d(256, 128, 1),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Conv1d(128, numpy_class, 1)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        x = self.layers(x)
        # x = F.log_softmax(x.view(-1,4), dim=-1)
        return x
        # return self.layers(x)


def batch_ohot_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=2)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=2)

    return y_pred_tags

def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    batch_size = features.size(0)

    mean_seg = None
    mean_seg_softmax = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=2)

    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            preds = models[MODEL_NUMBER](features.to(device=features.device)).transpose(2,1).contiguous()
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = batch_ohot_to_scalar(preds)
            img_seg = img_seg.reshape(*size).cpu().detach()
            if mean_seg_softmax is None:
                preds = F.log_softmax(preds.reshape(-1,4), dim=-1)
                preds = preds.reshape(batch_size, 2048, 4)
                mean_seg_softmax = preds
            else:
                preds = F.log_softmax(preds.reshape(-1,4), dim=-1)
                preds = preds.reshape(batch_size, 2048, 4)
                mean_seg_softmax += preds

            seg_mode_ensemble.append(img_seg)
        
        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        # top_k = js.sort()[0][:, - int(js.shape[1] / 10):].mean()
        top_k = torch.mean(js.sort()[0][:, - int(js.shape[1] / 10):],dim=1)

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1) # [2048, 10]
        img_seg_final = torch.mode(img_seg_final, 2)[0] # 
    return img_seg_final, top_k


def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred[0])

        mask = colorize_mask(pred[0], palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.jpg')
        )


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    print(args.model)
    if args.model == 'fold':
        save_dir = args.folding_classifier_path
    elif args.model == 'flow':
        save_dir = args.flow_classifier_path
    elif args.model == 'caps':
        save_dir = args.caps_classifier_path
    elif args.model == 'sgmae':
        save_dir = args.sgmae_classifier_path
    elif args.model == 'sgm':
        save_dir = args.sgm_classifier_path
    else:
        raise Exception(f"Wrong model type: {args.model}")
    models = []
    for i in range(args.model_total):
        model_path = os.path.join(save_dir, f'model_{i}.pth')
        state_dict = torch.load(model_path,map_location='cpu')
        model = pixel_classifier(args.num_class, args.dim[-1])
        model.load_state_dict(state_dict['model_state_dict'])
        model = nn.DataParallel(model, device_ids=[0,1,2])
        model = model.module.to(device)
        models.append(model)

    return models
