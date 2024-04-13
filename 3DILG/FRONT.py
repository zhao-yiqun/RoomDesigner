
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

# THREED_FRONT_LIVINGROOM_FURNITURE = {
#     "bookcase/jewelry armoire":                "bookshelf",
#     "desk":                                    "desk",
#     "pendant lamp":                            "pendant_lamp",
#     "ceiling lamp":                            "ceiling_lamp",
#     "lounge chair/cafe chair/office chair":    "lounge_chair",
#     "dining chair":                            "dining_chair",
#     "dining table":                            "dining_table",
#     "corner/side table":                       "corner_side_table",
#     "classic chinese chair":                   "chinese_chair",
#     "armchair":                                "armchair",
#     "shelf":                                   "shelf",
#     "sideboard/side cabinet/console table":    "console_table",
#     "footstool/sofastool/bed end stool/stool": "stool",
#     "barstool":                                "stool",
#     "round end table":                         "round_end_table",
#     "loveseat sofa":                           "loveseat_sofa",
#     "drawer chest/corner cabinet":             "cabinet",
#     "wardrobe":                                "wardrobe",
#     "three-seat/multi-seat sofa":              "multi_seat_sofa",
#     "wine cabinet":                            "wine_cabinet",
#     "coffee table":                            "coffee_table",
#     "lazy sofa":                               "lazy_sofa",
#     "children cabinet":                        "cabinet",
#     "chaise longue sofa":                      "chaise_longue_sofa",
#     "l-shaped sofa":                           "l_shaped_sofa",
#     "tv stand":                                "tv_stand"
# }
#
#
# FRONT_category = {
#     "bookshelf":0,
#     "desk":1,
#     "pendant_lamp":2,
#     "ceiling_lamp":3,
#     "lounge_chair":4,
#     "dining_chair":5,
#     "dining_table":6,
#     "corner_side_table":7,
#     "chinese_chair":8,
#     "armchair":9,
#     "shelf":10,
#     "console_table":11,
#     "stool":12,
#     "round_end_table":13,
#     "cabinet":14,
#     "wardrobe":15,
#     "multi_seat_sofa":16,
#     "wine_cabinet":17,
#     "coffee_table":18,
#     "lazy_sofa":19,
#     "sofa":20,
#     "chaise_longue_sofa":21,
#     "l_shaped_sofa":22,
#     "tv_stand":23
# }


# import clip
# _, preprocess = clip.load("ViT-B/32")


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



THREED_FRONT_LIVINGROOM_FURNITURE = {
    "bookcase/jewelry armoire":                "bookshelf",
    "desk":                                    "desk",
    "pendant lamp":                            "pendant_lamp",
    "ceiling lamp":                            "ceiling_lamp",
    "lounge chair/cafe chair/office chair":    "lounge_chair",
    "dining chair":                            "dining_chair",
    "dining table":                            "dining_table",
    "corner/side table":                       "corner_side_table",
    "classic chinese chair":                   "chinese_chair",
    "armchair":                                "armchair",
    "shelf":                                   "shelf",
    "sideboard/side cabinet/console table":    "console_table",
    "footstool/sofastool/bed end stool/stool": "stool",
    "barstool":                                "stool",
    "round end table":                         "round_end_table",
    "loveseat sofa":                           "loveseat_sofa",
    "drawer chest/corner cabinet":             "cabinet",
    "wardrobe":                                "wardrobe",
    "three-seat/multi-seat sofa":              "multi_seat_sofa",
    "wine cabinet":                            "wine_cabinet",
    "coffee table":                            "coffee_table",
    "lazy sofa":                               "lazy_sofa",
    "children cabinet":                        "cabinet",
    "chaise longue sofa":                      "chaise_longue_sofa",
    "l-shaped sofa":                           "l_shaped_sofa",
    "tv stand":                                "tv_stand"
}

Living_category = {
    "bookshelf":0,
    "desk":1,
    "pendant_lamp":2,
    "ceiling_lamp":3,
    "lounge_chair":4,
    "dining_chair":5,
    "dining_table":6,
    "corner_side_table":7,
    "chinese_chair":8,
    "armchair":9,
    "shelf":10,
    "console_table":11,
    "stool":12,
    "round_end_table":13,
    "loveseat_sofa":14,
    "cabinet":15,
    "wardrobe":16,
    "multi_seat_sofa":17,
    "wine_cabinet":18,
    "coffee_table":19,
    "lazy_sofa":20,
    "chaise_longue_sofa":21,
    "l_shaped_sofa":22,
    "tv_stand":23
}



# import clip
# _, preprocess = clip.load("ViT-B/32")

class _3DFRONT(data.Dataset):
    def __init__(self, dataset_folder, split, categories=None, transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, cloth_category='All'):

        self.cloth_category = cloth_category

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.dataset_folder = dataset_folder
        # self.dataset_folder = "/public/home/lijing1/"
        # self.dataset_folder = "/public/home/wuxh/"
        self.return_surface = return_surface
        self.surface_sampling = surface_sampling
        #
        self.point_folder = os.path.join(self.dataset_folder, '3D-FUTURE-model_sdf')
        # self.mesh_folder = os.path.join(self.dataset_folder, '3D-FUTURE_watertight')
        self.mesh_folder = os.path.join(self.dataset_folder, '3D-FUTURE-model_watertight')

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
                if item['category'] is not None:
                    u = item['category']
                    if '/' in u:
                        u = '/'.join([v.strip() for v in u.split('/')])
                    if u.lower() in THREED_FRONT_BEDROOM_FURNITURE.keys():
                        if item['model_id'] in ['193d183d-9a66-44a6-947e-db58f277f35d',
                                                '6289d90b-2b9f-4e68-bbfc-2314eeeb1dd4']: continue
                        self.file_paths += [os.path.join(self.mesh_folder, item['model_id'])]
                        self.models += [
                            {'category': THREED_FRONT_BEDROOM_FURNITURE[u.lower()],
                             'model': item['model_id']}
                        ]
            # import ipdb
            # ipdb.set_trace()
            categories = [cate_set]

        else:
            with open(os.path.join(self.point_folder, 'model_info.json'), 'r') as f:
                model_info = json.load(f)

            for item in model_info:

                if item['category'] is not None:
                    u = item['category']
                    if '/' in u:
                        u = '/'.join([v.strip() for v in u.split('/')])
                        # sp = THREED_FRONT_LIVINGROOM_FURNITURE[news]

                    if u.lower() in THREED_FRONT_LIVINGROOM_FURNITURE.keys() and THREED_FRONT_LIVINGROOM_FURNITURE[u.lower()] in categories:
                        if item['model_id'] in ['193d183d-9a66-44a6-947e-db58f277f35d',
                                                '6289d90b-2b9f-4e68-bbfc-2314eeeb1dd4']: continue
                        self.file_paths += [os.path.join(self.mesh_folder, item['model_id'])]
                        self.models += [
                            {'category': THREED_FRONT_LIVINGROOM_FURNITURE[u.lower()],
                             'model': item['model_id']}
                        ]
            # categories = [c for c in categories if
            #               os.path.isdir(os.path.join(self.point_folder, c)) and c.startswith('0')]
        categories.sort()
        # print(categories)
        if not self.split == 'train':
            # import ipdb
            # ipdb.set_trace()
            self.file_paths = self.file_paths[:5]
            self.models = self.models[:5]


        # if self.cloth_category == 'All':
        #     cur_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        #     for c_idx, c in enumerate(categories):
        #         subpath = os.path.join(
        #             cur_folder, c)
        #         assert os.path.isdir(subpath)
        #
        #         split_file = os.path.join(subpath, split + '.lst')
        #         with open(split_file, 'r') as f:
        #             models_c = f.read().split('\n')
        #
        #         self.models += [
        #             {'category': c, 'model': m.replace('.npz', '')}
        #             for m in models_c
        #         ]
        #         self.file_paths += [os.path.join(self.mesh_folder, c, m.replace('.npz', '')) for m in models_c]
        # else:
        #     print(f'TRAIN A SPECIFIC CATEGORY: {self.cloth_category}')
        #     for fp in (file_paths):
        #         if self.cloth_category in fp:
        #             self.file_paths.append(fp[:-1])

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

        # cur_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-model_sdf')
        #FUTURE_watertight

        # cur_file = self.file_paths[idx].replace('3D-FUTURE_watertight', '3D-FUTURE-model_sdf')
        # cur_file = self.file_paths[idx].replace('3D-FUTURE_watertight', '3D-FUTURE-model-points')
        cur_file = self.file_paths[idx].replace('3D-FUTURE-model_watertight', '3D-FUTURE-model-points')
        surface_path = f'{cur_file}_surface20w.npy'
        surface20w = np.load(surface_path)
        if not surface20w.shape == (200000, 3):
            import ipdb
            ipdb.set_trace()

        if self.surface_sampling:
            ind = np.random.default_rng().choice(surface20w.shape[0], self.pc_size, replace=False)
            surface = np.array(surface20w, dtype=np.float32)[ind]


        # images = torch.from_numpy(images)
        # depths = torch.from_numpy(depths)
        # condinfo = torch.from_numpy(condinfo)

        # return surface, images, depths, condinfo

        # '''
        #         Loading Query Points here.
        # '''
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
        surface = torch.from_numpy(surface)


        return points, labels, surface, FRONT_category[category] #, self.file_paths[idx].split('/')[-1]

        # return points, labels, surface, Living_category[category] , self.file_paths[idx].split('/')[-1]

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
#