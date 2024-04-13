import os
# import time
# import wget
# import shutil
# import torch
# import ocnn
import trimesh
import logging
import mesh2sdf
# import zipfile
import argparse
import numpy as np
from tqdm import tqdm
# from plyfile import PlyData, PlyElement

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=45572)
args = parser.parse_args()

size = 128         # resolution of SDF
level = 0.015      # 2/128 = 0.015625
shape_scale = 0.9  # rescale the shape into [-0.5, 0.5]
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# root_folder = '/storage/data/zhaoyq/ShapeNetCore.v2/'
# to_folder = '/public/home/zhaoyq/ShapeNetV2_watertight'
# to_folder1 = '/public/home/zhaoyq/ShapeNetV2_sdf'

to_folder = '/public/home/zhaoyq/3D-FUTURE-model_watertight'
to_folder1 = '/public/home/zhaoyq/3D-FUTURE-model_surf20w'
root_folder = '/storage/data/zhaoyq/3D-FUTURE-model'



# sa = []
# for root, dirs, files in os.walk(root_folder):
#         for filename in files:
#             if filename.endswith('raw_model.obj'):
#                 path = root.split('/')[-1]
#                 sa.append(path)
#                 # sa.append(','.join(path))
#
# with open("cad_front.csv", "w") as f:
#     for u in sa:
#         f.writelines(u + '\n')

def Sample_Surface(mesh, fp):
    surface, _ = trimesh.sample.sample_surface(mesh, 200000)
    npsurface = np.array(surface, dtype=np.float32)
    np.save(f'{fp}_surface20w.npy', npsurface)


# def run_mesh2sdf():
#     r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
#       '''
#
#     print('-> Run mesh2sdf.')
#     mesh_scale = 0.8
#     # filenames = get_filenames('unique_cads.csv')
#     with open('/public/home/zhaoyq/3D-FUTURE-model_sdf/None.csv') as f:
#         rows = [row for row in f]
#
#     f1 = open('nonormalized.csv', 'w')
#     f2 = open('novalid.csv', 'w')
#     for i in tqdm(range(args.start, args.end), ncols=80):
#         # u = rows[i].strip().split(',')
#         u = rows[i].strip()
#         # filename_raw = os.path.join(root_folder, u, 'raw_model.obj')
#         filename_raw = os.path.join(root_folder, u, 'normalized_model.obj')
#         if not os.path.exists(filename_raw):
#             print('No such normalized:  ', filename_raw)
#             f1.writelines(u + '\n')
#             filename_raw = os.path.join(root_folder, u, 'raw_model.obj')
#         filename_obj = os.path.join(to_folder, u + '.obj')
#         filename_npy = os.path.join(to_folder1, u)
#         try:
#             mesh = trimesh.load(filename_raw, force='mesh')
#         except:
#             print('No valid mesh: ',filename_raw)
#             f2.writelines(u + '\n')
#             continue
#
#         if not os.path.exists(os.path.join(to_folder1)):
#             os.makedirs(os.path.join(to_folder1))
#             os.makedirs(os.path.join(to_folder))
#         # rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
#         vertices = mesh.vertices
#         bbmin, bbmax = vertices.min(0), vertices.max(0)
#         center = (bbmin + bbmax) * 0.5
#         scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
#         vertices = (vertices - center) * scale
#         # run mesh2sdf
#         sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
#                                          level=level, return_mesh=True)
#         mesh_new.vertices = mesh_new.vertices * shape_scale
#         # save
#         # np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
#         # np.save(filename_npy, sdf)
#         # import ipdb
#         # ipdb.set_trace()
#         Sample_Surface(mesh, filename_npy)
#         mesh_new.export(filename_obj)
#
#     f1.close()
#     f2.close()


def run_mesh2sdf():
    r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
      '''

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    # filenames = get_filenames('unique_cads.csv')
    with open('/public/home/zhaoyq/ATISS/3DILG/cad_front.csv') as f:
        rows = [row for row in f]

    # f1 = open('nonormalized.csv', 'w')
    # f2 = open('novalid.csv', 'w')
    for i in tqdm(range(args.start, args.end), ncols=80):
        u = rows[i].strip()
        if u in [
            '193d183d-9a66-44a6-947e-db58f277f35d', '6289d90b-2b9f-4e68-bbfc-2314eeeb1dd4',
            '2ad973f5-ea6b-4b4e-85da-519ebc13fd17', '97526bba-32ef-4d19-8efd-6fd89dfd0610', '527bf14a-f606-4a59-a074-0bc17a322e11'
        ]:
            continue
        filename_raw = os.path.join(root_folder, u, 'my_normalized.obj')
        mesh = trimesh.load(filename_raw)
        mesh.vertices *= 1.9
        vertices = mesh.vertices
        sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
        filename_obj = os.path.join(to_folder, u + '.obj')
        mesh_new.export(filename_obj)
        # Sample_Surface(mesh_new, filename_npy)

    # f1.close()
    # f2.close()



# def main():
#     for root, _, files in os.walk('/public/home/zhaoyq/ShapeNetV2_watertight'):
#         for file in files:
#             if file.endswith('.obj'):
#                 to_path = os.path.join('/public/home/zhaoyq/ShapeNetV2_sdf', root.split('/')[-1], file[:-4])
#                 if os.path.exists(f'{to_path}_surface20w.npy'):
#                     continue
#                 print(to_path)
#                 mesh = trimesh.load(os.path.join(root, file), force="mesh", process=True)
#                 Sample_Surface(mesh, to_path)

def main():
    run_mesh2sdf()


main()