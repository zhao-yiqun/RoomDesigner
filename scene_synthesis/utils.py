# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#
import ipdb
import numpy as np
from PIL import Image
import trimesh
import torch
import math
import mcubes
import trimesh
from simple_3dviz.renderables.mesh import Mesh
import random
import cv2
# import nvdiffrast.torch as dr
import os
import seaborn as sns
# from .extract_texture_map import xatlas_uvmap
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh

#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import numpy as np
from PIL import Image
import trimesh

from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh



def get_textured_objects_with_specific(bbox_params_t, objects_dataset, classes,  fpath, idx, return_name=False):
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    names = []
    for j in range(1, bbox_params_t.shape[1] - 1):
        query_size = bbox_params_t[0, j, -4:-1]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )
        file_path = furniture.raw_model_path
        file_path = file_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/zhaoyq/dataused/')
        if idx +1 == j:
            file_path = fpath
            # import ipdb
            # ipdb.set_trace()
            # sfile = file_path.split('/')[-2]
            # file_path=file_path.replace(sfile, fpath)

        # Load the furniture and scale it as it is given in the dataset
        # file_path = furniture.raw_model_path
        # file_path = file_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/zhaoyq/dataused/')
        raw_mesh = Mesh.from_file(file_path)
        raw_mesh.scale(furniture.scale)
        if return_name:
            names.append(file_path)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1]) / 2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # updated version
        raw_mesh = Mesh.from_file(file_path)

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(file_path, force="mesh")
        texure_imgpath = furniture.texture_image_path.replace('/ibex/ai/home/parawr/data/scenegraphs/',
                                                              '/public/home/zhaoyq/dataused/')
        tr_mesh.visual.material.image = Image.open(
            texure_imgpath
        )
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes, names

def get_textured_objects(bbox_params_t, objects_dataset, classes, return_name=False):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    names = []
    for j in range(1, bbox_params_t.shape[1]-1):
        query_size = bbox_params_t[0, j, -4:-1]
        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        furniture = objects_dataset.get_closest_furniture_to_box(
            query_label, query_size
        )

        # Load the furniture and scale it as it is given in the dataset
        file_path = furniture.raw_model_path
        file_path = file_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/zhaoyq/dataused/')
        raw_mesh = Mesh.from_file(file_path)
        raw_mesh.scale(furniture.scale)
        if return_name:
            names.append(file_path)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # updated version
        raw_mesh = Mesh.from_file(file_path)



        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(file_path, force="mesh")
        texure_imgpath = furniture.texture_image_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/zhaoyq/dataused/')
        tr_mesh.visual.material.image = Image.open(
            texure_imgpath
        )
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes, names


def get_shaped_objects(bbox_params_t, objects_dataset, classes):
    # For each one of the boxes replace them with an object
    renderables = []
    lines_renderables = []
    trimesh_meshes = []
    shapes = bbox_params_t[..., -1280:]
    bbox_params_t = bbox_params_t[..., :-1280]
    for j in range(1, bbox_params_t.shape[1]-1):
        query_size = bbox_params_t[0, j, -4:-1]

        query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        query_shape = shapes[0, j]

        furniture = objects_dataset.get_closest_furniture_to_shape(
            query_label, query_shape, query_size
        )

        # Load the furniture and scale it as it is given in the dataset
        file_path = furniture.raw_model_path
        # import ipdb
        # ipdb.set_trace()
        file_path = file_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/wuxh/zhaoyq/data/')
        # file_path = file_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/zhaoyq/dataused/')
        raw_mesh = Mesh.from_file(file_path)
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        # updated version
        raw_mesh = Mesh.from_file(file_path)



        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R, t=translation)
        renderables.append(raw_mesh)

        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene
        tr_mesh = trimesh.load(file_path, force="mesh")
        texure_imgpath = furniture.texture_image_path.replace('/ibex/ai/home/parawr/data/scenegraphs/', '/public/home/wuxh/zhaoyq/data/')
        tr_mesh.visual.material.image = Image.open(
            texure_imgpath
        )
        tr_mesh.vertices *= furniture.scale
        tr_mesh.vertices -= centroid
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes

def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture)
    )

    tr_floor = trimesh.Trimesh(
        np.copy(vertices), np.copy(faces), process=False
    )
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture)
        )
    )

    return floor, tr_floor
r = lambda: random.randint(0, 255)



def latents_to_mesh(bbox_params_t, shapes, texture,  classes, vqvae_model, device='cuda', objects_dataset=None):
    renderables = []
    trimesh_meshes = []
    density: int = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density + 1)
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
    print(bbox_params_t.shape[1]-1)

    color_palette = np.array(sns.color_palette('hls', len(classes)))
    for j in range(1, bbox_params_t.shape[1]-1):

        shape_code = shapes[:, j]
        # centers = shape_code[..., :3]
        # centers_quantized.float()
        if shape_code[..., :3].max() > 10:
            centers = shape_code[..., :3].float() / 255.0 * 2 - 1
        else:
            # if objects_dataset is None:
            #     continue
            centers = shape_code[..., :3]
        entry = shape_code[..., 3]

        # ipdb.set_trace()
        # latents = vqvae_model.codebook_shape.get_codebook_entry(entry.long().cuda(), None)
        latents = vqvae_model.codebook.embedding(entry.long().cuda())
        N = 40000
        # import ipdb
        # ipdb.set_trace()
        logits = torch.cat([vqvae_model.decoder(latents, centers.cuda(), grid[:, i * N:(i + 1) * N])[0] for i in
                            range(math.ceil(grid.shape[1] / N))], dim=1)

        # logits = torch.cat([vqvae_model.decoder_shape(latents, centers.cuda(), grid[:, i * N:(i + 1) * N])[0] for i in
        #                     range(math.ceil(grid.shape[1] / N))], dim=1)
        volume = logits.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
        verts, faces = mcubes.marching_cubes(volume, 0)
        verts *= gap
        verts -= 1
        # 256 - > [-1, 1]
        # import ipdb.
        # ipdb.set_trace()
        shape_size = bbox_params_t[0, j, -4:-1]
        cur_size = (verts.max(0) - verts.min(0)) / 2
        scale = shape_size / cur_size
        centroid = (verts.max(0) + verts.min(0)) / 2
        verts = (verts - centroid) * scale

        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        # ipdb.set_trace()
        verts = verts.dot(R) + translation
        # shape_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
        class_index = bbox_params_t[0, j, :-7].argmax(-1)
        # color_palette = np.array(sns.color_palette('hls', len(classes)))
        rgb_color = color_palette[class_index, :]
        mesh = trimesh.Trimesh(verts, faces)
        mesh.visual.vertex_colors = np.tile(np.append(rgb_color, 1.0), (len(mesh.vertices), 1))
        trimesh_meshes.append(mesh)
        continue



        ctx = dr.RasterizeCudaContext(device=device)
        verts = torch.from_numpy(verts).cuda()
        faces_ = torch.from_numpy(faces.astype(np.int64)).cuda()
        uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(ctx, verts.float(), faces_, resolution=2048)
        tex_coord = gb_pos[mask.squeeze(3)]

        z_sample = torch.cat(torch.chunk(texture[:, j], 3, dim=1), dim=2)


        tex_output = torch.cat(
                [vqvae_model.decoder_texuture(tex_coord[None, i * N:(i + 1) * N], z_sample.permute(0, 2, 3, 1).cuda(), centers.cuda(), latents) for i in
                                     range(math.ceil(tex_coord.shape[0] / N))], dim=1)


        uv_out = torch.zeros_like(gb_pos)
        uv_out[mask.squeeze(3)] = tex_output


        shape_size = bbox_params_t[0, j, -4:-1]

        shape_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]




        #vertsp = verts * shape_size
        # we should add the texture first
        # FIXed:
        # The scale should not be the simply multiply.
        # It should be something like the de-scale.

        verts = verts.cpu().numpy()
        cur_size = (verts.max(0) - verts.min(0)) / 2
        scale = shape_size / cur_size
        centroid = (verts.max(0) + verts.min(0)) / 2
        vertsp = (verts - centroid) * scale

        translation = bbox_params_t[0, j, -7:-4]
        theta = bbox_params_t[0, j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        # ipdb.set_trace()
        vertsp = vertsp.dot(R) + translation

        # tr_mesh = trimesh.Trimesh(vertices=vertsp, faces=faces)
        # tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation



        lo, hi = (0, 1)
        # lo, hi = (-1, 1)
        img = np.asarray(uv_out[0].data.cpu().numpy(), dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = img.clip(0, 255)
        mask = np.sum(img.astype(np.float64), axis=-1, keepdims=True)
        mask = (mask <= 3.0).astype(np.float64)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img = img * (1 - mask) + dilate_img * mask
        img = img.clip(0, 255).astype(np.uint8)
        img = img[::-1]

        tr_mesh = (
            vertsp,
            uvs.data.cpu().numpy(),
            faces_.cpu().numpy(),
            mesh_tex_idx.cpu().numpy(),
            f"furniture_{j}.obj",
            img[..., ::-1],
            f"furniture_{j}.png"
        )

        # cv2.imwrite(f'{args.out_path}/{xyz[0]}.png', img[..., ::-1])
        # savemeshtes2(
        #     vertsp,
        #     uvs.data.cpu().numpy(),
        #     faces_.cpu().numpy(),
        #     mesh_tex_idx.cpu().numpy(),
        #     os.path.join(f'{args.out_path}/{xyz[0]}.obj')
        # )


        # rgb = np.array([r(), r(), r()])
        # raw_mesh = Mesh.from_faces(vertsp, faces, rgb)
        # mesh should be rescaled following the shape_size

        # raw_mesh.affine_transform(t=-centroid)
        # raw_mesh.affine_transform(R=R, t=translation)

        # renderables.append(raw_mesh)
        # Create a trimesh object for the same mesh in order to save
        # everything as a single scene

        # tr_mesh = trimesh.Trimesh(vertices = vertsp, faces = faces)
        # tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation

        trimesh_meshes.append(tr_mesh)

    return renderables, trimesh_meshes
        # raw_mesh = TexturedMesh.from_face(vqvae_model.raw_model_path)
        # print("shape-o", vqvae_model)

#
#
#
#
# def get_textured_objects(bbox_params_t, objects_dataset, classes):
#     # For each one of the boxes replace them with an object
#     renderables = []
#     trimesh_meshes = []
#     for j in range(1, bbox_params_t.shape[1]-1):
#         query_size = bbox_params_t[0, j, -4:-1]
#         query_label = classes[bbox_params_t[0, j, :-7].argmax(-1)]
#         furniture = objects_dataset.get_closest_furniture_to_box(
#             query_label, query_size
#         )
#
#         # Load the furniture and scale it as it is given in the dataset
#         raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
#         raw_mesh.scale(furniture.scale)
#
#         # Compute the centroid of the vertices in order to match the
#         # bbox (because the prediction only considers bboxes)
#         bbox = raw_mesh.bbox
#         centroid = (bbox[0] + bbox[1])/2
#
#         # Extract the predicted affine transformation to position the
#         # mesh
#         translation = bbox_params_t[0, j, -7:-4]
#         theta = bbox_params_t[0, j, -1]
#         R = np.zeros((3, 3))
#         R[0, 0] = np.cos(theta)
#         R[0, 2] = -np.sin(theta)
#         R[2, 0] = np.sin(theta)
#         R[2, 2] = np.cos(theta)
#         R[1, 1] = 1.
#
#         # Apply the transformations in order to correctly position the mesh
#         raw_mesh.affine_transform(t=-centroid)
#         raw_mesh.affine_transform(R=R, t=translation)
#         renderables.append(raw_mesh)
#
#         # Create a trimesh object for the same mesh in order to save
#         # everything as a single scene
#         tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
#         tr_mesh.visual.material.image = Image.open(
#             furniture.texture_image_path
#         )
#         tr_mesh.vertices *= furniture.scale
#         tr_mesh.vertices -= centroid
#         tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
#         trimesh_meshes.append(tr_mesh)
#
#     return renderables, trimesh_meshes


# def get_floor_plan(scene, floor_textures):
#     """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
#     TexturedMesh."""
#     vertices, faces = scene.floor_plan
#     vertices = vertices - scene.floor_plan_centroid
#     uv = np.copy(vertices[:, [0, 2]])
#     uv -= uv.min(axis=0)
#     uv /= 0.3  # repeat every 30cm
#     texture = np.random.choice(floor_textures)
#
#     floor = TexturedMesh.from_faces(
#         vertices=vertices,
#         uv=uv,
#         faces=faces,
#         material=Material.with_texture_image(texture)
#     )
#
#     tr_floor = trimesh.Trimesh(
#         np.copy(vertices), np.copy(faces), process=False
#     )
#     tr_floor.visual = trimesh.visual.TextureVisuals(
#         uv=np.copy(uv),
#         material=trimesh.visual.material.SimpleMaterial(
#             image=Image.open(texture)
#         )
#     )
#
#     return floor, tr_floor
