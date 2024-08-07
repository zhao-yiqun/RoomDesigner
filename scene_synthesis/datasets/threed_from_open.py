#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

from collections import Counter, OrderedDict
from functools import lru_cache
import numpy as np
import json
import os
import torch

from PIL import Image

from .common import BaseDataset
from .threed_front_scene import Room
from .utils import parse_threed_front_scenes


class ThreedFrontOpen(BaseDataset):
    """Container for the scenes in the 3D-FRONT dataset.

        Arguments
        ---------
        scenes: list of Room objects for all scenes in 3D-FRONT dataset
    """

    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None
        # self._shape_path = None

        self._sizes = self._centroids = self._angles = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
            len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        """The bbox for the entire dataset is simply computed based on the
        bounding boxes of all scenes in the dataset.
        """
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000] * 3)
        _size_max = np.array([-10000000] * 3)
        _centroid_min = np.array([10000000] * 3)
        _centroid_max = np.array([-10000000] * 3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])
        for s in self.scenes:
            for f in s.bboxes:
                if np.any(f.size > 5):
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)
        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)

    @property
    def bounds(self):
        return {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles
        }

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles

    @property
    def shape(self):
        if self._shape is None:
            self._shape = torch.load('tmp_path')
        return self._shape

    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(
            self.count_furniture.keys(),
            range(len(self.count_furniture))
        ))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum(
            [object_counts[k] for k, v in object_counts.items()]
        )
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types

    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        return self.object_types + ["start", "end"]

    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info,
                               path_to_models, path_to_room_masks_dir=None,
                               path_to_bounds=None, filter_fn=lambda s: s):
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir
        )
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)
        return cls([s for s in map(filter_fn, scenes) if s], bounds)


class CachedRoom(object):
    def __init__(
            self,
            scene_id,
            room_layout,
            floor_plan_vertices,
            floor_plan_faces,
            floor_plan_centroid,
            class_labels,
            translations,
            sizes,
            angles,
            image_path
    ):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]


class CachedRoomOpen(object):
    def __init__(
            self,
            scene_id,
            room_layout,
            floor_plan_vertices,
            floor_plan_faces,
            floor_plan_centroid,
            class_labels,
            translations,
            sizes,
            angles,
            image_path,
            jid_paths,
    ):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path
        self.jid_paths = jid_paths

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]


import random
# from timm.models import create_model
# from torch.multiprocessing import Manager


class CachedThreedFrontOpen(ThreedFrontOpen):
    def __init__(self, base_dir, config, scene_ids, splits):
        # super().__init__()
        self._base_dir = base_dir
        self.config = config

        self._parse_train_stats(config["train_stats"])

        self._tags = sorted([
            oi
            for oi in os.listdir(self._base_dir)
            if len(oi.split("_")) > 1 and oi.split("_")[1] in scene_ids
        ])
        self._path_to_rooms = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags
        ])
        # Load these in the CPU memory
        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(
            self._base_dir, self._tags[0], rendered_scene
        )
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        self._path_to_renders = sorted([
            os.path.join(self._base_dir, pi, rendered_scene)
            for pi in self._tags
        ])
        self.splits = splits
        # if "train" in self.splits:
        #     self.another_dict = Manager().dict()
        #     all_obj = set()
        #     for u in self._path_to_rooms:
        #         if u not in self.another_dict:
        #             D = np.load(u)
        #             shape_name = D["jids"]
        #             D = {key: D[key] for key in D.files}
        #             self.another_dict[u] = D
        #             for xv in shape_name:
        #                 all_obj.add(xv)
        #     self.cache_dict = Manager().dict()
        #
        #     # path = '/public/home/lijing1/3D-FUTURE-model-points'
        #     path = '/public/home/wuxh/3D-FUTURE-model-points'
        #     # dirs = os.listdir(path)
        #     for jid in all_obj:
        #         # jid = file.split("_")[0]
        #         if jid not in self.cache_dict:
        #             surface = np.load(os.path.join(path, f"{jid}_points.npy")).astype(np.float32)
        #             self.cache_dict[jid] = surface
        # for file in dirs:

        # self.img_dict = Manager().dict()

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(
            tuple(map(int, self.config["room_layout_size"].split(","))),
            resample=Image.BILINEAR
        )
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        # if "train" in self.splits:
        #     if self._path_to_rooms[i] in self.another_dict:
        #         D = self.another_dict[self._path_to_rooms[i]]
        #     else:
        #         assert False
        #         D = np.load(self._path_to_rooms[i])
        #         D = {key: D[key] for key in D.files}
        #         self.another_dict[self._path_to_rooms[i]] = D
        # else:
        #     D = np.load(self._path_to_rooms[i])
        # return D
        D = np.load(self._path_to_rooms[i])

        return CachedRoomOpen(
            scene_id=D["scene_id"],
            room_layout=self._get_room_layout(D["room_layout"]),
            floor_plan_vertices=D["floor_plan_vertices"],
            floor_plan_faces=D["floor_plan_faces"],
            floor_plan_centroid=D["floor_plan_centroid"],
            class_labels=D["class_labels"],
            translations=D["translations"],
            sizes=D["sizes"],
            angles=D["angles"],
            image_path=self._path_to_renders[i],
            jid_paths=D["jids"]
        )

    # Fixed: This should be added with a shape embedding
    def get_room_params(self, i):
        # if "train" in self.splits:
        #     if self._path_to_rooms[i] in self.another_dict:
        #         D = self.another_dict[self._path_to_rooms[i]]
        #     else:
        #         assert False
        #         D = np.load(self._path_to_rooms[i])
        #         D = {key: D[key] for key in D.files}
        #         self.another_dict[self._path_to_rooms[i]] = D
        # else:
        #     D = np.load(self._path_to_rooms[i])
        D = np.load(self._path_to_rooms[i])
        room = self._get_room_layout(D["room_layout"])
        room = np.transpose(room[:, :, None], (2, 0, 1))
        shape_name = D["jids"]
        surface_list = []
        for jid in shape_name:
            # if "train" in self.splits:
            #     if jid in self.cache_dict:
            #         surface20w = self.cache_dict[jid]
            #     else:
            #         assert False
            #         # surface20w_npy = os.path.join("/public/home/lijing1/3D-FUTURE-model-points", f"{jid}_points.npy")
            #         # surface20w_npy = os.path.join("/public/home/qianych/3D-FUTURE-model-points", f"{jid}_points.npy")
            #         surface20w_npy = os.path.join("/public/home/wuxh/3D-FUTURE-model-points", f"{jid}_points.npy")
            #         surface20w = np.load(surface20w_npy).astype(np.float32)
            #         self.cache_dict[jid] = surface20w
            # else:
            #     # surface20w_npy = os.path.join("/public/home/lijing1/3D-FUTURE-model-points", f"{jid}_points.npy")
            #     # surface20w_npy = os.path.join("/public/home/qianych/3D-FUTURE-model-points", f"{jid}_points.npy")
            #     surface20w_npy = os.path.join("/public/home/wuxh/3D-FUTURE-model-points", f"{jid}_points.npy")
            #     surface20w = np.load(surface20w_npy).astype(np.float32)
            # surface20w_npy = os.path.join("/public/home/zhaozb/3D-FUTURE-model_sdf", f"{jid}_surface20w.npy")
            # surface20w_npy = os.path.join("/public/home/zhaozb/3D-FUTURE-model-points", f"{jid}_surface20w.npy")
            # surface20w_npy = os.path.join("/public/home/qianych/3D-FUTURE-model-points", f"{jid}_surface20w.npy")
            # surface20w_npy = os.path.join("/public/home/zhaoyq/3D-FUTURE-model-points", f"{jid}_surface20w.npy")
            # surface20w_npy = os.path.join("/public/home/lijing1/3D-FUTURE-model-points", f"{jid}_surface20w.npy")
            # surface20w_npy = os.path.join("/public/home/wuxh/3D-FUTURE-model-points", f"{jid}_surface20w.npy")
            # surface20w = np.load(surface20w_npy).astype(np.float32)
            # ind = np.random.default_rng().choice(surface20w.shape[0], 20000, replace=False)
            # # surface = np.array(surface20w, dtype=np.float32)[ind]
            # surface = surface20w[ind]
            # ind = random.randint(0, 49)
            # surface20w_npy = os.path.join("/public/home/wuxh/3D-random-points", f"{jid}_points_{ind}.npy")
            # surface20w_npy = os.path.join("/public/home/qianych/3D-random-points", f"{jid}_points_{ind}.npy")
            # surface20w_npy = os.path.join("/public/home/zhaoyq/3D-random-points", f"{jid}_points_{ind}.npy")
            # surface20w_npy = os.path.join("/public/home/lijing1/3D-random-points", f"{jid}_points_{ind}.npy")
            # surface20w_npy = os.path.join("/public/home/zhaozb/3D-random-points", f"{jid}_points_{ind}.npy")
            # import ipdb
            # ipdb.set_trace()
            surface20w_npy = os.path.join("/public/home/huhzh/3D-FUTURE-model_shape_feat", f"{jid}_tensor.pt")

            shape_feat = torch.load(surface20w_npy, map_location="cpu")
            #= np.load(surface20w_npy).astype(np.float32)
            # ind = np.random.default_rng().choice(surface.shape[0], 2048, replace=False)
            # surface_list.append(surface[ind][None])
            surface_list.append(shape_feat.detach().numpy())
            # surface_list.append(surface[None])
        surface = np.concatenate((surface_list), axis=0)
        # shape embedding was encoded as surface points and using vae encode to encode it.
        return {
            "room_layout": room,
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"],
            "shapes": surface
        }

    def __len__(self):
        return len(self._path_to_rooms)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
            len(self), self.n_object_types
        )

    def _parse_train_stats(self, train_stats):
        # import ipdb
        # ipdb.set_trace()
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture
