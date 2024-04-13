
import os
import glob
import random

import yaml

import torch
from torch.utils import data

import numpy as np
import pandas as pd

import trimesh
import cv2

import mesh2sdf.core
import skimage.measure
import pysdf
import json

from PIL import Image

# Total Category {Top, Tshirt, Trousers, Skirt, Jumpsuit, Dress}



import torchvision.transforms as T



THREED_FRONT_BEDROOM_FURNITURE = {
    "desk":                                    "desk",
    "nightstand":                              "nightstand",
    "king-size bed":                           "double_bed",
    "single bed":                              "single_bed",
    "kids bed":                                "kids_bed",
    "ceiling lamp":                            "ceiling_lamp",
    "pendant lamp":                            "pendant_lamp",
    "bookcase/jewelry armoire":                "bookshelf",
    "tv stand":                                "tv_stand",
    "wardrobe":                                "wardrobe",
    "lounge chair/cafe chair/office chair":    "chair",
    "dining chair":                            "chair",
    "classic chinese chair":                   "chair",
    "armchair":                                "armchair",
    "dressing table":                          "dressing_table",
    "dressing chair":                          "dressing_chair",
    "corner/side table":                       "table",
    "dining table":                            "table",
    "round end table":                         "table",
    "drawer chest/corner cabinet":             "cabinet",
    "sideboard/side cabinet/console table":    "cabinet",
    "children cabinet":                        "children_cabinet",
    "shelf":                                   "shelf",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",#????
    "coffee table":                            "coffee_table",
    "loveseat sofa":                           "sofa",
    "three-seat/multi-seat sofa":              "sofa",
    "l-shaped sofa":                           "sofa",
    "lazy sofa":                               "sofa",
    "chaise longue sofa":                      "sofa",
}

FRONT_category = {
    "desk":0,
    "nightstand":1,
    "double_bed":2,
    "single_bed":3,
    "kids_bed":4,
    "ceiling_lamp":5,
    "pendant_lamp":6,
    "bookshelf":7,
    "tv_stand":8,
    "wardrobe":9,
    "chair":10,
    "armchair":11,
    "dressing_table":12,
    "dressing_chair":13,
    "table":14,
    "cabinet":15,
    "children_cabinet":16,
    "shelf":17,
    "stool":18,
    "coffee_table":19,
    "sofa":20,
}


# import clip
# _, preprocess = clip.load("ViT-B/32")

class Dmtet_3DFRONT(data.Dataset):
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

        self.point_folder = os.path.join(self.dataset_folder  , '3D-FUTURE-model_sdf')
        self.mesh_folder = os.path.join(self.dataset_folder  , '3D-FUTURE-model_watertight')


        # storage_path = '/public/home/qianych'

        # storage_path = '/data'
        #
        # self.point_folder = os.path.join(self.dataset_folder , '3D-FUTURE-model_sdf')
        # self.mesh_folder = os.path.join(storage_path , '3D-FUTURE-model_watertight')

        # storage_path = '/storage/group/gaoshh'
        #
        # self.point_folder = os.path.join(storage_path , '3D-FUTURE-model_sdf')
        # self.mesh_folder = os.path.join(storage_path , '3D-FUTURE-model_watertight')

        # if self.split == 'train':
        #     self.file_list_path = f'{self.dataset_folder}/WatertightTriMeshTrain.txt'
        # else:
        #     self.file_list_path = f'{self.dataset_folder}/TriMeshTest_t1.txt'

        # file_list = open(self.file_list_path, "r")
        # file_paths = file_list.readlines()
        self.models = []
        self.file_paths = []
        if categories is None:
            # categories = os.listdir(self.point_folder)
            cate_set = set()
            for u in THREED_FRONT_BEDROOM_FURNITURE.values():
                cate_set.add(u)
            with open(os.path.join(self.point_folder, 'model_info.json'), 'r') as f:
                model_info = json.load(f)

            for item in model_info:
                if item['category'] is not None and item['category'].lower() in THREED_FRONT_BEDROOM_FURNITURE.keys():
                    if item['model_id'] in ['193d183d-9a66-44a6-947e-db58f277f35d', '6289d90b-2b9f-4e68-bbfc-2314eeeb1dd4',
                                            '2ad973f5-ea6b-4b4e-85da-519ebc13fd17',
                                            '97526bba-32ef-4d19-8efd-6fd89dfd0610',
                                            '527bf14a-f606-4a59-a074-0bc17a322e11']:continue
                    self.file_paths += [os.path.join(self.mesh_folder, item['model_id'])]
                    self.models += [
                        {'category': THREED_FRONT_BEDROOM_FURNITURE[item['category'].lower()],
                         'model': item['model_id']}
                    ]

            categories = [cate_set]
            # categories = [c for c in categories if
            #               os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        # print(categories)
        if not self.split == 'train':
            self.file_paths = self.file_paths[:5]
            self.models = self.models[:5]

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
        # watertight_mesh = trimesh.load(f'{self.file_paths[idx]}.obj',
        #                                force="mesh",
        #                                process=True)

        # f = pysdf.SDF(watertight_mesh.vertices, watertight_mesh.faces)

        '''
        Loading Surafece Point here.
        '''
        # cur_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-points')
        cur_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-model_surf20w')
        # cur_file = cur_file.replace('/storage/data/zhaoyq/', '/public/home/zhaoyq/')
        img_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-renders')
        # img_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-DA')
        mni = img_file.split('/')[:-1]
        mni[0] = f'/{mni[0]}'
        img_file = os.path.join(*mni,  model)
        camera_file = os.path.join(*mni, model, 'info.json')

        masks = []
        depths = []
        # transformation = None
        ind = np.random.default_rng().choice(20, 8, replace=False)

        with open(camera_file) as f:
            cameras = json.load(f)

        img_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-comp')
        depth_mask = cv2.imread(f'{img_file}_comp.png', flags=cv2.IMREAD_UNCHANGED)

        condinfo = []
        for i in range(8):
            filed = "%05d" % ind[i]
            # print(depth_mask.shape)
            masks.append(depth_mask[1024 * ind[i]: 1024*(ind[i]+1), :1024])
            depths.append(depth_mask[1024 * ind[i]: 1024*(ind[i]+1), 1024:])
            # masks.append(cv2.imread(os.path.join(img_file, f'{filed}_a.png'), flags=cv2.IMREAD_UNCHANGED))
            # depths.append(cv2.imread(os.path.join(img_file, f'{filed}_depth.png'), flags=cv2.IMREAD_UNCHANGED).astype(np.float32))
            camera_trans = cameras['frames'][ind[i]][f'{filed}frame']
            transforms = np.eye(4)
            # Due to the Grimbel lock, we can not use angle to change it.
            # So we change to use transformation matrix to represent it.
            transforms[:3, 0] = np.array(camera_trans['x'])
            transforms[:3, 1] = np.array(camera_trans['y'])
            transforms[:3, 2] = np.array(camera_trans['z'])
            transforms[:3, 3] = np.array(camera_trans['origin'])
            transforms = transforms.transpose()
            transforms[:, 1], transforms[:, 2] = transforms[:, 2], -transforms[:, 1]
            R = transforms[:3, :3]
            T = transforms[3, :3]
            RT = np.column_stack((R, -1*R@T))
            transforms = np.row_stack((RT, np.array([0, 0, 0, 1])))
            condinfo.append(transforms)


        condinfo = np.array(condinfo)


        masks = np.array(masks)
        background = np.zeros_like(masks)
        foreground = np.ones_like(masks)
        images = foreground * ( masks > 0) + background * (1-(masks > 0))
        images = images.astype(np.float32)
        # images[images > 32768] = 1
        scale = 6.5/65535
        depths = np.array(depths) * scale
        depths[depths == 6.5] = 0 # This is to make equaivelent with OpenGL rendered depth
        depths = -depths


        # condinfo = np.zeros((2, 8))
        # rotation_camera = np.load(os.path.join(camera_file, 'rotation.npy'))
        # elevation_camera = np.load(os.path.join(camera_file, 'elevation.npy'))
        # condinfo[0] = rotation_camera / 180 * np.pi
        # condinfo[1] = (90 - elevation_camera) / 180 * np.pi

        # surface_path = f'{cur_file}_points.npy'
        surface_path = f'{cur_file}_surface20w.npy'
            #os.path.join(cur_file, 'points.npy')
        surface = np.load(surface_path)


        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
            surface = np.array(surface[..., :3], dtype=np.float32)[ind]
        # surface[:, 0] ,surface[:, 1], surface[:, 2] = surface[:, 0], -surface[:, 1], -surface[:, 2]
        surface = torch.from_numpy(surface)
        images = torch.from_numpy(images)
        depths = torch.from_numpy(depths)
        condinfo = torch.from_numpy(condinfo)

        return surface, images, depths, condinfo , model

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