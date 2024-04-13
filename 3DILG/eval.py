import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vqvae_512_1024_2048', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument('--pth', default='output/vqvae_512_1024_2048_Front//checkpoint-799.pth', type=str)
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_path', default='/public/home/zhaoyq' , required=True, type=str,
                    help='dataset path')
args = parser.parse_args()

from tqdm import tqdm 

import yaml

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as T

import numpy as np

from scipy.spatial import cKDTree as KDTree

import trimesh
import mcubes

import modeling_vqvae
from shapenet import ShapeNet, category_ids
from FRONT import _3DFRONT, FRONT_category
from timm.models import create_model
import utils

from pathlib import Path

def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = create_model(
        args.model,
    )
    device = torch.device(args.device)

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
    model.to(device)


    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

    with torch.no_grad():
        
        metric_loggers = []
        for category, _ in FRONT_category.items():
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_loggers.append(metric_logger)
            header = 'Test:'

            dataset_test = _3DFRONT(args.data_path, split='train', transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=2048)
            # dataset_test = ShapeNet(args.data_path, split='test', categories=[category], transform=None, sampling=False, return_surface=True, surface_sampling=False)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=1,
                num_workers=8,
                drop_last=False,
            )

            for batch in metric_logger.log_every(data_loader_test, 10, header):
                points, labels, surface, _ = batch

                ind = np.random.default_rng().choice(surface[0].numpy().shape[0], 2048, replace=False)

                surface2048 = surface[0][ind][None]

                surface2048 = surface2048.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                N = 50000

                _, latents, centers_quantized, _, _, _ = model.encode(surface2048)
                centers = centers_quantized.float() / 255.0 * 2 - 1

                output = torch.cat([model.decoder(latents, centers, points[:, i*N:(i+1)*N])[0] for i in range(math.ceil(grid.shape[1]/N))], dim=1)

                pred = torch.zeros_like(output[0])
                pred[output[0]>=0] = 1
                intersection = (pred * labels[0]).sum()
                union = (pred + labels[0]).gt(0).sum()
                iou = intersection * 1.0 / union

                metric_logger.update(iou=iou.item())

                output = torch.cat([model.decoder(latents, centers, grid[:, i*N:(i+1)*N])[0] for i in range(math.ceil(grid.shape[1]/N))], dim=1)

                volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0)
                verts *= gap
                verts -= 1.
                m = trimesh.Trimesh(verts, faces)
                
                pred = m.sample(100000)

                tree = KDTree(pred)
                dist, _ = tree.query(surface[0].cpu().numpy())
                d1 = dist
                gt_to_gen_chamfer = np.mean(dist)
                gt_to_gen_chamfer_sq = np.mean(np.square(dist))

                tree = KDTree(surface[0].cpu().numpy())
                dist, _ = tree.query(pred)
                d2 = dist
                gen_to_gt_chamfer = np.mean(dist)
                gen_to_gt_chamfer_sq = np.mean(np.square(dist))

                cd = gt_to_gen_chamfer + gen_to_gt_chamfer

                metric_logger.update(cd=cd)

                th = 0.02

                if len(d1) and len(d2):
                    recall = float(sum(d < th for d in d2)) / float(len(d2))
                    precision = float(sum(d < th for d in d1)) / float(len(d1))

                    if recall+precision > 0:
                        fscore = 2 * recall * precision / (recall + precision)
                    else:
                        fscore = 0
                metric_logger.update(fscore=fscore)

            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg)

        print(args)
        for (category, _), metric_logger in zip(category_ids.items(), metric_loggers):
            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg)

if __name__ == '__main__':
    main()