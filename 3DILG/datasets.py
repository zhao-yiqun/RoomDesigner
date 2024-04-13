import os

import torch
from torchvision import datasets, transforms

# from shapenet import ShapeNet
# from shapenet_own import ShapeNet_watertight
from FRONT import _3DFRONT
# from Front_dmtet import Dmtet_3DFRONT

from Front_texture import FTRONT_tex
class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        # print(scaling)
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point


class AxisScaling_texture(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, point, tex_points):
        scale = torch.rand(1) * 0.5 + 0.5
        surface[..., :3] *= scale
        point[..., :3] *= scale
        tex_points[..., :3] *= scale
        # surface[..., :3] += np.random.normal(scale=0.005, size=(n_near_points, 3))
        # scaling = torch.rand(1, 3) * 0.5 + 0.75
        # # print(scaling)
        # surface[..., :3] = surface[..., :3] * scaling
        # point = point * scaling
        # tex_points[..., :3] = tex_points[..., :3] * scaling
        #
        # scale = (1 / torch.abs(surface[..., :3]).max().item()) * 0.999999
        # surface[..., :3] *= scale
        # point *= scale
        # tex_points[..., :3] *= scale

        # if self.jitter:
        #     surface[..., :3] += 0.005 * torch.randn_like(surface[..., :3])
        #     surface[..., :3].clamp_(min=-1, max=1)

        return surface, point, tex_points


class AxisScaling_origin(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, points):
        scale = torch.rand(1) * 0.5 + 0.5
        surface[..., :3] *= scale
        points[..., :3] *= scale
        # tex_points[..., :3] *= scale
        # surface[..., :3] += np.random.normal(scale=0.005, size=(n_near_points, 3))
        # scaling = torch.rand(1, 3) * 0.5 + 0.75
        # # print(scaling)
        # surface[..., :3] = surface[..., :3] * scaling
        # point = point * scaling
        # tex_points[..., :3] = tex_points[..., :3] * scaling
        #
        # scale = (1 / torch.abs(surface[..., :3]).max().item()) * 0.999999
        # surface[..., :3] *= scale
        # point *= scale
        # tex_points[..., :3] *= scale

        # if self.jitter:
        #     surface[..., :3] += 0.005 * torch.randn_like(surface[..., :3])
        #     surface[..., :3].clamp_(min=-1, max=1)
        return surface, points
        # return surface, point, tex_points

# def build_shape_surface_occupancy_dataset(split, args):
#     if split == 'train':
#         transform = AxisScaling_origin((0.75, 1.25), True)
#         # transform = AxisScaling_texture((0.75, 1.25), True)
#         return FTRONT_tex(args.data_path, split=split, points_path="3D-FUTURE-model-points", transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     elif split == 'val':
#         return FTRONT_tex(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     else:
#         return FTRONT_tex(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)

# def build_shape_surface_occupancy_dataset(split, args):
#     if split == 'train':
#         transform = AxisScaling((0.75, 1.25), True)
#         return Dmtet_3DFRONT(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     elif split == 'val':
#         return Dmtet_3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     else:
#         return Dmtet_3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)


# def build_shape_surface_occupancy_dataset(split, args):
#     if split == 'train':
#         transform = AxisScaling((0.75, 1.25), True)
#         return _3DFRONT(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     elif split == 'val':
#         return _3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
#     else:
#         return _3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)



def build_shape_surface_occupancy_dataset(split, args):
    if split == 'train':
        transform = AxisScaling((0.75, 1.25), True)
        return _3DFRONT(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        return _3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return _3DFRONT(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)


if __name__ == '__main__':
    pass