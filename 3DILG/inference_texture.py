import math

import argparse

import ipdb
#/public/home/qianych/zhaoyq/3DILG/output/vqvae_512_1024_2048_new/checkpoint-399.pth
#/public/home/zhaoyq/ATISS/3DILG/output/vqvae_512_1024_2048_Front/checkpoint-6999.pth
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='vqvae_512_1024_2048', type=str,
                    metavar='MODEL', help='Name of model to train')
parser.add_argument('--pth', default='/public/home/qianych/zhaoyq/3DILG/output/vqvae_512_1024_2048_new/checkpoint-499.pth', type=str)
parser.add_argument('--device', default='cuda:0',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--data_path', default='/public/home/zhaoyq', type=str,
                    help='dataset path')
parser.add_argument('--out_path', default='test_feape_cross_1499', type=str,
                    help='dataset path')
parser.add_argument('--coord_pe',  action='store_true', help='whether to add the coords pe')

args = parser.parse_args()

from tqdm import tqdm

import yaml

import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import nvdiffrast.torch as dr
import torchvision.transforms as T

import cv2
import numpy as np

from scipy.spatial import cKDTree as KDTree

import trimesh
import mcubes
from extract_texture_map import xatlas_uvmap, savemeshtes2
# import modeling_vqvae_texture
# import modeling_vqvae_mix
import modeling_vqvae
from shapenet import ShapeNet, category_ids
# from FRONT import _3DFRONT, FRONT_category
from Front_texture import  FTRONT_tex, FRONT_category
from timm.models import create_model
import utils
import ipdb
from pathlib import Path


# Whether train from 2w while inference from 2048 is ok.
# Whether train from k=64 and test from 32 is ok.

def main():
    print(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = create_model(
        args.model, c_dim = 32, hidden_dim = 32, n_blocks=3,
        affine_size = 24, reso=128, aware=0, plane_min = 0, coord_pe= args.coord_pe,
    )
    device = torch.device(args.device)

    # ipdb.set_trace()

    model.eval()
    model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density + 1)
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
    st = torch.tensor([41]).cuda()
    with torch.no_grad():

        metric_loggers = []
        for category, _ in FRONT_category.items():
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_loggers.append(metric_logger)
            header = 'Test:'

            dataset_test = FTRONT_tex(args.data_path, split='eval',  points_path='3D-FUTURE-model-points', transform=None, sampling=False, return_surface=True,
                                    surface_sampling=True, pc_size=2048)
            # import ipdb
            # ipdb.set_trace()
            # dataset_test.file_paths = [dataset_test.file_paths[42] for _ in range(50)]
            # dataset_test.models = [dataset_test.models[42] for _ in range(50)]
            # dataset_test = ShapeNet(args.data_path, split='test', categories=[category], transform=None, sampling=False, return_surface=True, surface_sampling=False)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=1,
                num_workers=8,
                drop_last=False,
            )
            # ipdb.set_trace()
            ind = 0
            for batch in metric_logger.log_every(data_loader_test, 10, header):
                # points, labels, surface, _, xyz = batch
                points, labels, surface, _, surface_tex_in, surface_tex_out, xyz = batch
                # import ipdb
                # ipdb.set_trace()

                # ind = np.random.default_rng().choice(surface[0].numpy().shape[0], 2048, replace=False)

                # surface2048 = surface[0][ind][None]
                surface2048 = surface
                surface2048 = surface2048.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # ind = np.random.default_rng().choice(surface_tex_in[0].numpy().shape[0], 2048, replace=False)

                surface_tex_in = surface_tex_in.to(device, non_blocking=True)
                surface_tex_out = surface_tex_out.to(device, non_blocking=True)

                N = 50000

                # _, latents, centers_quantized, _, _, _ = model.encode_shape(surface2048, surface2048)
                _, latents, centers_quantized, _, _, encodings = model.encode(surface2048)
                centers = centers_quantized.float() / 255.0 * 2 - 1
                #
                # #
                # z_sample, kld = model.encode_texture(surface2048, surface_tex_in)
                # #

                # import ipdb
                # ipdb.set_trace()
                st = torch.cat((st, encodings.unique()))
                # #

                output = torch.cat(
                    [model.decoder(latents, centers, points[:, i * N:(i + 1) * N])[0] for i in
                     range(math.ceil(grid.shape[1] / N))], dim=1)
                # output = torch.cat(
                #     [model.decoder_shape(latents, centers, points[:, i * N:(i + 1) * N])[0] for i in
                #      range(math.ceil(grid.shape[1] / N))], dim=1)
                #
                #
                # # z_sample, kld = model.encode(surface2048, surface_tex_in)
                #
                # # ipdb.set_trace()
                # # evals, output = model.decoder(surface_tex_out[..., :3], z_sample, points)
                #
                # # pred = torch.zeros_like(output)
                # # pred[output >= 0] = 1
                # # intersection = (pred * labels[0]).sum()
                # # union = (pred + labels[0]).gt(0).sum()
                # # iou = intersection * 1.0 / union
                #
                pred = torch.zeros_like(output[0])
                pred[output[0] >= 0] = 1
                intersection = (pred * labels[0]).sum()
                union = (pred + labels[0]).gt(0).sum()
                iou = intersection * 1.0 / union
                #
                # metric_logger.update(iou=iou.item())
                # # ipdb.set_trace()
                #
                output = torch.cat([model.decoder(latents, centers, grid[:, i * N:(i + 1) * N])[0] for i in
                                    range(math.ceil(grid.shape[1] / N))], dim=1)
                # # ipdb.set_trace()
                # output = torch.cat([model.decoder(points, z_sample, grid[:, i * N:(i + 1) * N])[1]
                #                     for i in range(math.ceil(grid.shape[1]/N))
                #                     ], dim=1)
                #
                #
                volume = output.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
                verts, faces = mcubes.marching_cubes(volume, 0)
                verts *= gap
                # ipdb.set_trace()
                verts -= 1.
                m = trimesh.Trimesh(verts, faces)




                # _, latents, centers_quantized, _, _, _, _ = model.encode(surface2048, surface_tex_in)
                # centers = centers_quantized.float() / 255.0 * 2 - 1
                m.export(f'test_new/{xyz[0]}.obj')
                # m.export(f'test_old/{xyz[0]}_{ind}.obj')
                # ind += 1
                continue


                # import ipdb
                # ipdb.set_trace()

                # output = torch.cat([model.decoder.get_surface(latents, centers, points[:, i * N:(i + 1) * N])[0] for i in
                #                     range(math.ceil(points.shape[1] / N))], dim=1)
                #
                # pred = torch.zeros_like(output[0])
                # pred[output[0] >= 0] = 1
                # intersection = (pred * labels[0]).sum()
                # union = (pred + labels[0]).gt(0).sum()
                # iou = intersection * 1.0 / union
                #
                # metric_logger.update(iou=iou.item())
                # # ipdb.set_trace()
                # output = torch.cat([model.decoder.get_surface(latents, centers,  grid[:, i * N:(i + 1) * N])[0] for i in
                #                     range(math.ceil(grid.shape[1] / N))], dim=1)
                #
                # volume = output.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
                # verts, faces = mcubes.marching_cubes(volume, 0)
                # verts *= gap
                # verts -= 1.

                # m = trimesh.Trimesh(verts, faces)

                ctx = dr.RasterizeCudaContext(device=device)
                verts = torch.from_numpy(verts).to(device)
                faces_ = torch.from_numpy(faces.astype(np.int64)).to(device)
                uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(ctx, verts.float(), faces_, resolution=2048)
                tex_coord = gb_pos[mask.squeeze(3)]

                # tex_output = torch.cat([model.decoder_texuture(tex_coord[None, i * N:(i + 1) * N], z_sample, centers, latents) for i in
                #                     range(math.ceil(tex_coord.shape[0] / N))], dim=1)

                # tex_output = torch.cat(
                #     [model.decoder_texuture(tex_coord[None, i * N:(i + 1) * N], z_sample) for i in
                #                          range(math.ceil(tex_coord.shape[0] / N))], dim=1)
                # # ipdb.set_trace()
                #
                # evals = torch.cat(
                #     [model.decoder_texuture(surface_tex_out[:, i * N:(i + 1) * N][..., :3], z_sample)
                #      for i in range(math.ceil(surface_tex_out.shape[1] / N))], dim=1)

                # evals = torch.cat([model.decoder_texuture(surface_tex_out[:, i * N:(i + 1) * N][..., :3], z_sample, centers, latents) for i in
                                    # range(math.ceil(surface_tex_out.shape[1] / N))], dim=1)

                # print(tex_coord.shape)
                # N = 30000
                # ipdb.set_trace()
                # tex_output = torch.cat(
                #     [model.decoder(tex_coord[None, i*N:(i+1)*N][..., :3], z_sample, points)[0]
                #      for i in range(math.ceil(tex_coord.shape[0]/N)) ], dim=1
                # )
                # ipdb.set_trace()
                tex_output = torch.cat([
                    model.decoder.get_texture(latents, centers, tex_coord[None, i * N:(i + 1) * N][..., :3])[0] for i in
                    range(math.ceil(tex_coord.shape[0] / N))], dim=1)

                evals = torch.cat([
                    model.decoder.get_texture(latents, centers, surface_tex_out[:, i * N:(i + 1) * N][..., :3])[0] for i in
                    range(math.ceil(surface_tex_out.shape[1]/N))], dim=1)


                criterion = torch.nn.MSELoss()
                mse = criterion(evals, surface_tex_out[..., 3:])

                metric_logger.update(mse=mse)
                uv_out = torch.zeros_like(gb_pos)
                uv_out[mask.squeeze(3)] = tex_output
                # ipdb.set_trace()
                savemeshtes2(
                    verts.cpu().numpy(),
                    uvs.data.cpu().numpy(),
                    faces_.cpu().numpy(),
                    mesh_tex_idx.cpu().numpy(),
                    os.path.join(f'{args.out_path}/{xyz[0]}.obj')
                )
                # import ipdb
                # ipdb.set_trace()

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
                cv2.imwrite(f'{args.out_path}/{xyz[0]}.png', img[..., ::-1])


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

                    if recall + precision > 0:
                        fscore = 2 * recall * precision / (recall + precision)
                    else:
                        fscore = 0
                metric_logger.update(fscore=fscore)
                # m.export(f'tested_4000/{xyz[0]}.obj')

            print(len(st.unique()))
            print(st.unique())

            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg, metric_logger.mse.avg)
            break

        print(args)
        for (category, _), metric_logger in zip(category_ids.items(), metric_loggers):
            print(category, metric_logger.iou.avg, metric_logger.cd.avg, metric_logger.fscore.avg)


if __name__ == '__main__':
    main()