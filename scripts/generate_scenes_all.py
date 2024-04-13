#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used for generating scenes using a previously trained model."""
import argparse
import logging
import os
import sys

import cv2
import ipdb
import numpy as np
import torch

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects, latents_to_mesh
# from scene_synthesis.networks.vqvae_model import vqvae_512_1024_2048 as vqvae_model
# from scene_synthesis.networks.vqvae_model import vqvae_256_1024_2048 as vqvae_model
from scene_synthesis.networks.vqvae_model import vqvae_128_1024_2048 as vqvae_model
from scene_synthesis.networks.vqvae_model_texture import vqvae_512_1024_2048_cross as vqvae_tex
# from simple_3dviz import Scene
# from simple_3dviz.window import show
# from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
# from simple_3dviz.behaviours.misc import LightToCamera
# from simple_3dviz.behaviours.movements import CameraTrajectory
# from simple_3dviz.behaviours.trajectory import Circle
# from simple_3dviz.behaviours.io import SaveFrames, SaveGif
# from simple_3dviz.utils import render
from scene_synthesis.extract_texture_map import savemeshtes2

bbox = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, 1, -1],
    [1, 1, 1],
])
from copy import deepcopy

def iou_3d(box1, box2):
    x1, y1, z1, w1, h1, d1 = box1
    x2, y2, z2, w2, h2, d2 = box2

    x_overlap = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
    y_overlap = max(0, min(y1 + h1/2, y2 + h2/2) - max(y1 - h1/2, y2 - h2/2))
    z_overlap = max(0, min(z1 + d1/2, z2 + d2/2) - max(z1 - d1/2, z2 - d2/2))

    intersection_volume = x_overlap * y_overlap * z_overlap
    volume1 = w1 * h1 * d1
    volume2 = w2 * h2 * d2

    union_volume = volume1 + volume2 - intersection_volume

    iou = intersection_volume / union_volume
    return iou
def create_bbox(verts, bbox_params, ind):
    cur_size = (verts.max(0) - verts.min(0)) / 2
    shape_size = bbox_params[0, ind, -4:-1]
    scale = shape_size / cur_size
    verts_j = verts * scale
    translation = bbox_params[0, ind, -7:-4]
    theta = bbox_params[0, ind, -1]
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(theta)
    R[0, 2] = -np.sin(theta)
    R[2, 0] = np.sin(theta)
    R[2, 2] = np.cos(theta)
    R[1, 1] = 1.
    verts_j = verts_j.dot(R) + translation
    return verts_j
def nms_bbox(bbox_params, bbox_shapes):
    verts = deepcopy(bbox)
    bbox_params_s = [bbox_params[:, 0:1]]
    bbox_shapes_s = [bbox_shapes[:, 0:1]]
    for j in range(1, bbox_params.shape[1]-1):
        bbox_j = create_bbox(verts, bbox_params, j)
        iou_a = bbox_j.min(0)
        vol_lena = bbox_j.max(0) - bbox_j.min(0)
        iou_j = np.concatenate((iou_a, vol_lena), axis=0)
        ok = 0
        for i in range(j+1, bbox_params.shape[1]-1):
            bbox_i = create_bbox(verts, bbox_params, i)
            iou_b = bbox_i.min(0)
            vol_lenb = bbox_i.max(0) - bbox_i.min(0)
            iou_i = np.concatenate((iou_b, vol_lenb), axis=0)
            ious = iou_3d(iou_j, iou_i)
            if ious >= 0.65:
                ok=1
        if ok:
            continue
        bbox_params_s.append(bbox_params[:, j:j+1])
        bbox_shapes_s.append(bbox_shapes[:, j:j+1])

    SLEN = bbox_params.shape[1]-1
    bbox_params_s.append(bbox_params[:, SLEN:SLEN+1])
    bbox_shapes_s.append(bbox_shapes[:, SLEN:SLEN+1])

    return np.concatenate(bbox_params_s, axis=1), torch.cat(bbox_shapes_s, dim=1)





def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        default="./tmp/",
        help="Path to the output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=1000,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )
    parser.add_argument(
        "--begin",
        default=0,
        type=int,
        help="The scene id to be used for conditioning"
    )
    parser.add_argument(
        "--end",
        default=1000,
        type=int,
        help="The scene id to be used for conditioning"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))
    # ipdb.set_trace()
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Create the scene and the behaviour list for simple-3dviz
    # scene = Scene(size=args.window_size)
    # scene.up_vector = args.up_vector
    # scene.camera_target = args.camera_target
    # scene.camera_position = args.camera_position
    # scene.light = args.camera_position

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    classes = np.array(dataset.class_labels)

    # import ipdb
    # ipdb.set_trace()
    if "all" in config["network"].get("type") or "shape" in config["network"].get("type"):
        vqvae = vqvae_model(pretrained=False, **config["pretrain_params"]).to(device)
        # vqvae = vqvae_tex(pretrained=False, **config["pretrain_params"]).to(device)
        pretraind_path = config["network"].get("pretrained_path")
        vqvae.load_state_dict(torch.load(pretraind_path, map_location='cpu')['model'], strict=True)

    # given_scene_id = 28
    floor_id = [31, 34, 35, 37, 45, 66, 83, 84, 115, 119, 136, 156, 161, 197, 205]
    # for i in range(args.n_sequences):
    for given_scene_id in floor_id:
        for i in range(20):
            scene_idx = given_scene_id

            current_scene = raw_dataset[scene_idx]
            print("{} / {}: Using the {} floor plan of scene {}".format(
                i, args.n_sequences, scene_idx, current_scene.scene_id)
            )
            # Get a floor plan
            floor_plan, tr_floor, room_mask = floor_plan_from_scene(
                current_scene, args.path_to_floor_plan_textures
            )

            bbox_params = network.generate_boxes(
                room_mask=room_mask.to(device),
                device=device
            )

            boxes = dataset.post_process(bbox_params)
            # ipdb.set_trace()
            bbox_params_t = torch.cat([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"],
                # boxes["shapes"]
            ], dim=-1).cpu().numpy()
            # print(bbox_params_t.shape[1])
            # bbox_params_t , boxes["shapes"] = nms_bbox(bbox_params_t, boxes["shapes"])
            # print(bbox_params_t.shape[1])

            # renderables, trimesh_meshes = get_textured_objects(
            #     bbox_params_t, objects_dataset, classes
            # )

            with torch.no_grad():
                renderables, trimesh_meshes = latents_to_mesh(
                    bbox_params_t, boxes["shapes"], boxes["shapes"], classes, vqvae, objects_dataset
                )
            renderables += floor_plan
            trimesh_meshes += tr_floor

            if trimesh_meshes is not None:
                # Create a trimesh scene and export it
                path_directory_lis = args.output_directory.split('/')
                path_directory_lis[-1] = f'bedroom_{given_scene_id}'
                dir = '/'.join(path_directory_lis)
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path_to_objs = os.path.join(
                    dir,
                    "{:03d}_scene".format(i)
                )
                if not os.path.exists(path_to_objs):
                    os.mkdir(path_to_objs)
                np.save(os.path.join(path_to_objs, 'box_params.npy'), bbox_params_t)
                export_scene(path_to_objs, trimesh_meshes)


if __name__ == "__main__":
    main(sys.argv[1:])
