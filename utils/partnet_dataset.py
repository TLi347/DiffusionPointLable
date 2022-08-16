'''
    Dataset for ShapeNetPart segmentation
'''

import os
import os.path
import json
from typing import cast
from cv2 import PCA_USE_AVG
from matplotlib.pyplot import axis
import numpy as np
import sys
import random
import point_cloud_utils as pcu

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PartNetDataset():
    def __init__(self, root, npoints = 2500, split='train', normalize=True, cates='chair', k_shot=-1):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.k_shot = k_shot
        
        self.normalize = normalize
        
        
        self.meta = []
        dir_point = os.path.join(self.root, cates+'_'+split)
        fns = sorted(os.listdir(dir_point))
        #print(os.path.basename(fns))
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0]) 
            self.meta.append(os.path.join(dir_point, token))
        
        self.datapath = []
        for fn in self.meta:
            self.datapath.append((fn))
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        # self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            data = os.path.join(self.datapath[index], 'pts_10000.ply')
            point_set = pcu.load_mesh_v(data).astype(np.float32)
            if self.normalize:
                point_set = pc_normalize(point_set)
            else:
                shift = point_set.mean(axis=0).reshape(1, 3)
                scale = point_set.flatten().std().reshape(1, 1)
                point_set = (point_set - shift) / scale
            seg = np.loadtxt(os.path.join(self.datapath[index], 'pts_10000_label.txt')).astype(np.int32)
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        return point_set, seg
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = PartNetDataset(root = '/data2/tli/seg-diffusion-point-cloud/data', split='train', npoints=2048, normalize=False, cates='chair')
    print('len(d) = ', len(d))

    import point_cloud_utils as pcu
    segs = []
    for i in range(1000):
        ps, seg = d[i]
        segs.append(seg)
    segs = np.concatenate(segs,axis=-1)
    print(np.unique(segs))
        

    
    # # sys.path.append('../utils')
    # # import show3d_balls
    # # show3d_balls.showpoints(ps, normal+1, ballradius=8)

    # d = PartNormalDataset(root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal', classification = True)
    # print('classification true')
    # print(d.cat)
    # print(len(d))
    # ps, normal, cls = d[0]
    # print(ps.shape, type(ps), cls.shape, type(cls))

    # d = PartNormalDataset(root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal', classification = False)
    # print('classification true')
    # print(d.cat)
    # print(len(d))
    # ps, normal, cls = d[0]
    # print(ps.shape, type(ps), cls.shape, type(cls))

    