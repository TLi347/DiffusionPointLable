import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type == 'ddpm':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type == 'swav':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    elif model_type == 'deeplab':
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


class FeatureDataset(Dataset):
    ''' 
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self, 
        X_data: torch.Tensor, 
        y_data: torch.Tensor
    ):    
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)




def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class InMemoryPCLabelDataset(Dataset):
    ''' 

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3. 

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            pcs: np.ndarray,
            labels: np.ndarray,
            num_points=2048,
            normarlize=True
    ):
        super().__init__()
        assert  len(pcs) == len(labels)
        self.pcs = pcs
        self.labels = labels
        self.num_points = num_points
        self.normarlize = normarlize

    def __len__(self):
        return len(self.pcs)

    def __getitem__(self, idx):
        pc = self.pcs[idx]
        assert pc.shape[0] == self.num_points, \
            f"Only 2048 points are supported"
        if self.normarlize:
            pc = pc_normalize(pc)
        tensor_pc = torch.from_numpy(pc).to(dtype=torch.float32)
        label = self.labels[idx]
        tensor_label = torch.from_numpy(label)
        
        return tensor_pc, tensor_label




