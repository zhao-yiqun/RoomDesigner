import os
import glob
import random

import yaml

import torch
from torch.utils import data

import numpy as np
import pandas as pd

import trimesh

import mesh2sdf.core
import skimage.measure
import pysdf

from PIL import Image

# Total Category {Top, Tshirt, Trousers, Skirt, Jumpsuit, Dress}


category_ids = {
    '02691156': 0,
    '02747177': 1,
    '02773838': 2,
    '02801938': 3,
    '02808440': 4,
    '02818832': 5,
    '02828884': 6,
    '02843684': 7,
    '02871439': 8,
    '02876657': 9,
    '02880940': 10,
    '02924116': 11,
    '02933112': 12,
    '02942699': 13,
    '02946921': 14,
    '02954340': 15,
    '02958343': 16,
    '02992529': 17,
    '03001627': 18,
    '03046257': 19,
    '03085013': 20,
    '03207941': 21,
    '03211117': 22,
    '03261776': 23,
    '03325088': 24,
    '03337140': 25,
    '03467517': 26,
    '03513137': 27,
    '03593526': 28,
    '03624134': 29,
    '03636649': 30,
    '03642806': 31,
    '03691459': 32,
    '03710193': 33,
    '03759954': 34,
    '03761084': 35,
    '03790512': 36,
    '03797390': 37,
    '03928116': 38,
    '03938244': 39,
    '03948459': 40,
    '03991062': 41,
    '04004475': 42,
    '04074963': 43,
    '04090263': 44,
    '04099429': 45,
    '04225987': 46,
    '04256520': 47,
    '04330267': 48,
    '04379243': 49,
    '04401088': 50,
    '04460130': 51,
    '04468005': 52,
    '04530566': 53,
    '04554684': 54,
}

import torchvision.transforms as T


# import clip
# _, preprocess = clip.load("ViT-B/32")

class ShapeNet_watertight(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, cloth_category='All'):

        self.cloth_category = cloth_category

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_sdf')
        self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')
        # if self.split == 'train':
        #     self.file_list_path = f'{self.dataset_folder}/WatertightTriMeshTrain.txt'
        # else:
        #     self.file_list_path = f'{self.dataset_folder}/TriMeshTest_t1.txt'

        # file_list = open(self.file_list_path, "r")
        # file_paths = file_list.readlines()
        if categories is None:
            categories = os.listdir(self.point_folder)
            categories = [c for c in categories if
                          os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        # print(categories)

        self.models = []
        self.file_paths = []
        if self.cloth_category == 'All':
            cur_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
            for c_idx, c in enumerate(categories):
                subpath = os.path.join(
                    cur_folder, c)
                assert os.path.isdir(subpath)

                split_file = os.path.join(subpath, split + '.lst')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')

                self.models += [
                    {'category': c, 'model': m.replace('.npz', '')}
                    for m in models_c
                ]
                self.file_paths += [os.path.join(self.mesh_folder, c, m.replace('.npz', '')) for m in models_c]
        else:
            print(f'TRAIN A SPECIFIC CATEGORY: {self.cloth_category}')
            for fp in (file_paths):
                if self.cloth_category in fp:
                    self.file_paths.append(fp[:-1])

        self.low_bound = 0.
        self.up_bound = 0.

        self.z_shift = np.array([0., 0., 0.2], dtype=np.float32)

        self.error_surface = []

    def __getitem__(self, idx):

        # category = f'{self.file_paths[idx]}'.split('/')[-1]
        # import ipdb
        # ipdb.set_trace()
        category = self.models[idx]['category']
        model = self.models[idx]['model']

        # print(self.file_paths[idx])

        '''
        Loading Watertight Mesh and SDF Function here.
        '''
        watertight_mesh = trimesh.load(f'{self.file_paths[idx]}.obj',
                                       force="mesh",
                                       process=True)

        f = pysdf.SDF(watertight_mesh.vertices, watertight_mesh.faces)

        '''
        Loading Surafece Point here.
        '''
        cur_file = self.file_paths[idx].replace('ShapeNetV2_watertight', 'ShapeNetV2_sdf')
        surface_path = f'{cur_file}_surface20w.npy'
        surface20w = np.load(surface_path)
        # print(surface_path)
        if not surface20w.shape == (200000, 3):
            import ipdb
            ipdb.set_trace()

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface20w.shape[0], self.pc_size, replace=False)
            surface = np.array(surface20w, dtype=np.float32)[ind]
            # surface = surface + self.z_shift
        surface = torch.from_numpy(surface)

        '''
        Loading Query Points here.
        '''
        # Init Near Points here.
        n_near_points = self.num_samples // 2
        ind = np.random.default_rng().choice(surface20w.shape[0], n_near_points, replace=False)
        surface_points = np.array(surface20w, dtype=np.float32)[ind]
        near_points = np.concatenate([
            surface_points + np.random.normal(scale=0.005, size=(n_near_points, 3)),
            surface_points + np.random.normal(scale=0.05, size=(n_near_points, 3)),
        ], axis=0)
        near_sdf = f(near_points)
        near_label = near_sdf >= 0
        near_points = near_points.astype(np.float32)
        near_points = torch.from_numpy(near_points)
        near_label = torch.from_numpy(near_label)

        # Init Volume Points here.
        n_uniform_points = 128 ** 3
        vol_points = np.random.rand(n_uniform_points, 3) * 2 - 1
        ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
        vol_points = vol_points[ind]
        vol_sdf = f(vol_points)
        vol_label = vol_sdf >= 0
        vol_points = vol_points.astype(np.float32)
        vol_points = torch.from_numpy(vol_points)
        vol_label = torch.from_numpy(vol_label)

        points = torch.cat([vol_points, near_points], dim=0)
        labels = torch.cat([vol_label, near_label], dim=0).float()

        return points, labels, surface, category_ids[category]#, idx #, cloth_static[category]

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    pass

# n_uniform_points = args.num_volume_points
#     n_near_points = args.num_near_points // 2

#     vol_points = np.random.rand(n_uniform_points, 3) * 2 - 1

#     surface_points = scaled_mesh.sample(n_near_points)
#     near_points = np.concatenate([
#         surface_points + np.random.normal(scale=0.005, size=(n_near_points, 3)),
#         surface_points + np.random.normal(scale=0.05, size=(n_near_points, 3)),
#     ], axis=0)

#     sdf = pysdf.SDF(scaled_mesh.vertices, scaled_mesh.faces)

#     vol_sdf = sdf(vol_points)
#     near_sdf = sdf(near_points)

#     vol_label = vol_sdf >= 0
#     near_label = near_sdf >= 0

#     vol_points = vol_points.astype(np.float32)
#     near_points = near_points.astype(np.float32)