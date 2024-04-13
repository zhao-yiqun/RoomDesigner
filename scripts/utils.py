# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import os
import pickle

import numpy as np
import torch
from PIL import Image
from pyrr import Matrix44

import trimesh

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.utils import save_frame
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.utils import render as render_simple_3dviz

from scene_synthesis.utils import get_textured_objects, latents_to_mesh#, get_textured_objects_with_specific


class DirLock(object):
    def __init__(self, dirpath):
        self._dirpath = dirpath
        self._acquired = False

    @property
    def is_acquired(self):
        return self._acquired

    def acquire(self):
        if self._acquired:
            return
        try:
            os.mkdir(self._dirpath)
            self._acquired = True
        except FileExistsError:
            pass

    def release(self):
        if not self._acquired:
            return
        try:
            os.rmdir(self._dirpath)
            self._acquired = False
        except FileNotFoundError:
            self._acquired = False
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def ensure_parent_directory_exists(filepath):
    os.makedirs(filepath, exist_ok=True)


def floor_plan_renderable(room, color=(1.0, 1.0, 1.0, 1.0)):
    vertices, faces = room.floor_plan
    # Center the floor
    vertices -= room.floor_plan_centroid
    # Return a simple-3dviz renderable
    return Mesh.from_faces(vertices, faces, color)


def floor_plan_from_scene(
    scene,
    path_to_floor_plan_textures,
    without_room_mask=False
):
    if not without_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_mask = None
    # Also get a renderable for the floor plan
    floor, tr_floor = get_floor_plan(
        scene,
        [
            os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)
        ]
    )
    return [floor], [tr_floor], room_mask


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    # floor = TexturedMesh.from_faces(
    #     vertices=vertices,
    #     uv=uv,
    #     faces=faces,
    #     material=Material.with_texture_image(texture)
    # )
    # floor = Mesh.from_faces(
    #     vertices=vertices,
    #     # uv = uv,
    #     faces=faces,
    #     colors=np.array([1.0, 1.0, 1.0])
    # )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )


    tr_floor.export('tmp.obj')
    floor = Mesh.from_file(
        'tmp.obj'
    )
    # floor = tr_floor

    if not texture.endswith("png"):
        # print('yes!', texture)
        texture += '/texture.png'
    # print(texture)
    # tr_floor.visual = trimesh.visual.TextureVisuals(
    #     uv=np.copy(uv),
    #     material=trimesh.visual.material.SimpleMaterial(
    #         image=Image.open(texture)
    #     )
    # )

    return floor, tr_floor


def get_textured_objects_in_scene(scene, ignore_lamps=False):
    renderables = []
    for furniture in scene.bboxes:
        model_path = furniture.raw_model_path
        if not model_path.endswith("obj"):
            import pdb
            pdb.set_trace()

        if not os.path.exists(furniture.raw_model_path):
            import pdb
            pdb.set_trace()

        #print(furniture.raw_model_path)
        # Load the furniture and scale it as it is given in the dataset
        raw_mesh = TexturedMesh.from_file(model_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)
    return renderables


def render(scene, renderables, color, mode, frame_path=None):
    if color is not None:

        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size, background=args.background)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position
    scene.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )
    return scene


def export_scene(output_directory, trimesh_meshes, names=None):
    if names is None:
        names = [
            "object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))
        ]
    mtl_names = [
        "material_{:03d}".format(i) for i in range(len(trimesh_meshes))
    ]

    for i, m in enumerate(trimesh_meshes):
        # obj_out, tex_out = trimesh.exchange.obj.export_obj(
        #     m,
        #     return_texture=True
        # )

        obj_out = trimesh.exchange.obj.export_obj(
            m
        )
        tex_out = None

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(
                    b"material0", mtl_names[i].encode("ascii")
                )
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def print_predicted_labels(dataset, boxes):
    object_types = np.array(dataset.object_types)
    box_id = boxes["class_labels"][0, 1:-1].argmax(-1)
    labels = object_types[box_id.cpu().numpy()].tolist()
    print("The predicted scene contains {}".format(labels))


def poll_specific_class(dataset):
    label = input(
        "Select an object class from {}\n".format(dataset.object_types)
    )
    if label in dataset.object_types:
        return dataset.object_types.index(label)
    else:
        return None


def make_network_input(current_boxes, indices, device):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices])
    )


def make_network_input_shape(current_boxes, indices, device):
    def _prepare(x):
        return torch.from_numpy(x[None].astype(np.float32))

    return dict(
        class_labels=_prepare(current_boxes["class_labels"][indices]),
        translations=_prepare(current_boxes["translations"][indices]),
        sizes=_prepare(current_boxes["sizes"][indices]),
        angles=_prepare(current_boxes["angles"][indices]),
        shapes=_prepare(current_boxes["shapes"][indices])
    )


def render_to_folder(
    args,
    folder,
    dataset,
    objects_dataset,
    tr_floor,
    floor_plan,
    scene,
    bbox_params,
    file_path=None,
    idx_out=None,
    add_start_end=False,
):
    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu()

    if add_start_end:
        bbox_params_t = torch.cat([
            torch.zeros(1, 1, bbox_params_t.shape[2]),
            bbox_params_t,
            torch.zeros(1, 1, bbox_params_t.shape[2]),
        ], dim=1)

    # renderables, trimesh_meshes, names = get_textured_objects_with_specific(bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels),  file_path, idx_out, True)
    #
    renderables, trimesh_meshes, names = get_textured_objects(
        bbox_params_t.numpy(), objects_dataset, np.array(dataset.class_labels), True
    )
    trimesh_meshes += tr_floor

    path_to_objs = os.path.join(args.output_directory, folder)
    if not os.path.exists(path_to_objs):
        os.mkdir(path_to_objs)
    export_scene(path_to_objs, trimesh_meshes)

    path_to_image = os.path.join(
        args.output_directory,
        folder + "_render.png"
    )
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image, 1)
    ]
    render_simple_3dviz(
        renderables + floor_plan,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )


def render_scene_from_bbox_params(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    path_to_image,
    path_to_objs
):

    boxes = dataset.post_process(bbox_params)

    # import ipdb
    # ipdb.set_trace()
    print_predicted_labels(dataset, boxes)
    # bbox_params_t = torch.cat(
    #     [
    #         torch.from_numpy(boxes["class_labels"]),
    #         torch.from_numpy(boxes["translations"]),
    #         torch.from_numpy(boxes["sizes"]),
    #         torch.from_numpy(boxes["angles"])
    #     ],
    #     dim=-1
    # ).cpu().numpy()

    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"],
        ],
        dim=-1
    ).cpu().numpy()
    renderables, trimesh_meshes, names = get_textured_objects(
        bbox_params_t, objects_dataset, classes, True
    )
    renderables += floor_plan
    trimesh_meshes += tr_floor

    # Do the rendering
    behaviours = [
        LightToCamera(),
        SaveFrames(path_to_image+".png", 1)
    ]
    render_simple_3dviz(
        renderables,
        behaviours=behaviours,
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        n_frames=args.n_frames,
        scene=scene
    )

    # import ipdb
    # ipdb.set_trace()
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)
        np.save(os.path.join(path_to_objs, 'box_params.npy'), bbox_params_t)
        # import ipdb
        # ipdb.set_trace()
        save_files = [u.split("/")[-2] for u in names]
        # for u in names:
        #     save_files.append(u.split("/")[-2])
        with open(os.path.join(path_to_objs, "object_name.pkl"), "wb") as f:
            pickle.dump(save_files, f)


        export_scene(path_to_objs, trimesh_meshes)



def render_scene_from_bbox_params_shape(
    args,
    bbox_params,
    dataset,
    objects_dataset,
    classes,
    floor_plan,
    tr_floor,
    scene,
    path_to_image,
    path_to_objs,
    vqvae
):



    boxes = dataset.post_process(bbox_params)
    print_predicted_labels(dataset, boxes)


    bbox_params_t = torch.cat(
        [
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ],
        dim=-1
    ).cpu().numpy()
    print("begin decoding")
    # pre_x = 3
    # for i in range(1, boxes["shapes"].shape[1]-1):
    #     if boxes["shapes"][:, i, ..., :3].max()>2:
    #         pre_x = i
    #         break
    # 前面pre_x个去retrieve 其他的自己生成 Bedroom-8445

    # import ipdb
    # ipdb.set_trace()
    # import ipdb
    # ipdb.set_trace()
    # renderables, trimesh_meshes, names = get_textured_objects(
    #     bbox_params_t, objects_dataset, classes, False
    # )

    with torch.no_grad():
        renderables, trimesh_meshes = latents_to_mesh(
            bbox_params_t, boxes["shapes"], boxes["shapes"], classes, vqvae, objects_dataset
        )


    # import ipdb
    # ipdb.set_trace()

    # renderables, trimesh_meshes = get_textured_objects(
    #     bbox_params_t, objects_dataset, classes
    # )
    # renderables += floor_plan

    # Merge all the data
    # for u in renderables2:
    #     renderables.append(u)
    #
    # for u in trimesh_meshes2:
    #     trimesh_meshes.append(u)
    trimesh_meshes += tr_floor

    # import ipdb
    # ipdb.set_trace()
    # Do the rendering
    # behaviours = [
    #     LightToCamera(),
    #     SaveFrames(path_to_image+".png", 1)
    # ]
    # render_simple_3dviz(
    #     renderables,
    #     behaviours=behaviours,
    #     size=args.window_size,
    #     camera_position=args.camera_position,
    #     camera_target=args.camera_target,
    #     up_vector=args.up_vector,
    #     background=args.background,
    #     n_frames=args.n_frames,
    #     scene=scene
    # )
    if trimesh_meshes is not None:
        # Create a trimesh scene and export it
        if not os.path.exists(path_to_objs):
            os.mkdir(path_to_objs)

        np.save(os.path.join(path_to_objs, 'shapes_anchor.npy'), boxes['shapes'].numpy())
        np.save(os.path.join(path_to_objs, 'box_params.npy'), bbox_params_t)
        export_scene(path_to_objs, trimesh_meshes)

