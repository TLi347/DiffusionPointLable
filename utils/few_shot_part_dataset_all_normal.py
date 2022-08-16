'''
    Dataset for ShapeNetPart segmentation
'''

import os
import os.path
import json
from typing import cast
from matplotlib.pyplot import axis
import numpy as np
import sys
import random

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class FewShotPartNormalDataset():
    def __init__(self, root, npoints = 2500, classification = False, split='train', normalize=True, return_cls_label = False, cates='chair', k_shot=-1):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.k_shot = k_shot
        
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in self.cat.items()}
        if cates == 'chair':
            self.cat = {'Chair':'03001627'}#12
        if cates == 'airplane':
            self.cat = {'Airplane':'02691156'}#0
        if cates == 'guitar':
            self.cat = {'Guitar':'03467517'}#19-21
        if cates == 'lamp':
            self.cat = {'Lamp':'03636649'}#24-27
        if cates == 'table':
            self.cat = {'Table':'04379243'}#47-49
        if cates == 'bag':
            self.cat = {'Bag':'02773838'}#4-5
        if cates == 'car':
            self.cat = {'Car':'02958343'}#8-11
        if cates == 'mug':
            self.cat = {'Mug':'03797390'}#36-37
        # self.cat = {'Chair':	'03001627'}
        # self.cat = {'Motorbike': '03790512'}
        #print(self.cat)
            
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-4])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids) or (fn[0:-4] in test_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            if self.k_shot > 0 and len(fns) > self.k_shot:
                fns = random.sample(fns, self.k_shot) # random few-shot samples                
                pass

            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            else:
                shift = point_set.mean(axis=0).reshape(1, 3)
                scale = point_set.flatten().std().reshape(1, 1)
                point_set = (point_set - shift) / scale
            normal = data[:,3:6]
            seg = data[:,-1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)
                
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice,:]
        if self.classification:
            return point_set, normal, cls
        else:
            if self.return_cls_label:
                return point_set, normal, seg, cls
            else:
                return point_set, normal, seg-12
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = FewShotPartNormalDataset(root = '/data2/tli/ae-diffusion-point-cloud/data/shapenetcore_partanno_segmentation_benchmark_v0_normal', split='trainval', npoints=3000, normalize=False, cates='chair')
    print('len(d) = ', len(d))

    # import point_cloud_utils as pcu
    # for i in range(5):
    #     ps, normal, seg = d[i]
    #     print(seg.min(), seg.max())
        

    
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

    