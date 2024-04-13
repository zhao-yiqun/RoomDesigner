import torch
import numpy as np
import os
import mcubes
from modeling_vqvae import vqvae_512_1024_2048 as model_vqvae
import math
import trimesh
from tqdm import tqdm

root_folder = '/public/home/zhaoyq/ATISS/3DILG/tested_4000'
vqvae = model_vqvae()
vqvae.load_state_dict(torch.load('/public/home/zhaoyq/pre_ckpt/checkpoint-5999.pth', map_location='cpu')['model'], strict=True)
vqvae = vqvae.cuda()
density = 128
gap = 2. / density
x = np.linspace(-1, 1, density + 1)
y = np.linspace(-1, 1, density + 1)
z = np.linspace(-1, 1, density + 1)
xv, yv, zv = np.meshgrid(x, y, z)
grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()

def sort(centers_quantized, encodings):
    ind3 = torch.argsort(centers_quantized[:, :, 0], dim=1)
    centers_quantized = torch.gather(centers_quantized, 1, ind3[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind3[:, :, None].expand(-1, -1, encodings.shape[-1]))

    _, ind2 = torch.sort(centers_quantized[:, :, 2], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind2[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind2[:, :, None].expand(-1, -1, encodings.shape[-1]))

    _, ind1 = torch.sort(centers_quantized[:, :, 1], dim=1, stable=True)
    centers_quantized = torch.gather(centers_quantized, 1, ind1[:, :, None].expand(-1, -1, centers_quantized.shape[-1]))
    encodings = torch.gather(encodings, 1, ind1[:, :, None].expand(-1, -1, encodings.shape[-1]))
    return centers_quantized, encodings

def get_mixture_mesh(file1, file2):
    surf1 = np.load(file1)
    surf2 = np.load(file2)
    ind1 = np.random.default_rng().choice(surf1.shape[0], 2048, replace=False)
    ind2 = np.random.default_rng().choice(surf2.shape[0], 2048, replace=False)

    surf1 = torch.from_numpy(surf1[ind1]).cuda()
    surf2 = torch.from_numpy(surf2[ind2]).cuda()
    _, z_q_x_st_1, centers_quantized1, _, _, _ = vqvae.encode(surf1[None])
    _, z_q_x_st_2, centers_quantized2, _, _, _ = vqvae.encode(surf2[None])
    centers_quantized1_s, encodings1_s = sort(centers_quantized1, z_q_x_st_1)
    centers_quantized2_s, encodings2_s = sort(centers_quantized2, z_q_x_st_2)
    mix_centers1 = torch.cat([centers_quantized1_s[:, :256], centers_quantized2_s[:, 256:]], dim=1)
    mix_centers2 = torch.cat([centers_quantized2_s[:, :256], centers_quantized1_s[:, 256:]], dim=1)
    mix_encodings1 = torch.cat([encodings1_s[:, :256], encodings2_s[:, 256:]], dim=1)
    mix_encodings2 = torch.cat([encodings2_s[:, :256], encodings1_s[:, 256:]], dim=1)
    mix_centers1 = mix_centers1/255.0*2 - 1
    mix_centers2 = mix_centers2/255.0*2 - 1
    N = 80000

    output1 = torch.cat([vqvae.decoder(mix_encodings1, mix_centers1, grid[:, i * N:(i + 1) * N])[0] for i in
                        range(math.ceil(grid.shape[1] / N))], dim=1)

    output2 = torch.cat([vqvae.decoder(mix_encodings2, mix_centers2, grid[:, i * N:(i + 1) * N])[0] for i in
                                    range(math.ceil(grid.shape[1] / N))], dim=1)

    volume1 = output1.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()
    volume2 = output2.view(density + 1, density + 1, density + 1).permute(1, 0, 2).cpu().numpy()

    verts1, faces1 = mcubes.marching_cubes(volume1, 0)
    verts1 *= gap
    verts1 -= 1.
    verts2, faces2 = mcubes.marching_cubes(volume2, 0)
    verts2 *= gap
    verts2 -= 1.
    mesh1 = trimesh.Trimesh(verts1, faces1)
    mesh2 = trimesh.Trimesh(verts2, faces2)
    return mesh1, mesh2



filenames = os.listdir(root_folder)
for i in tqdm(range(len(filenames))):
    for j in range(i+1, len(filenames)):
        u = filenames[i].split('.')[0]
        v = filenames[j].split('.')[0]
        file1 = os.path.join('/public/home/zhaoyq/3D-FUTURE-model-points',
                             f"{u}_surface20w.npy")
        file2 = os.path.join('/public/home/zhaoyq/3D-FUTURE-model-points',
                             f"{v}_surface20w.npy")

        with torch.no_grad():
            mesh1, mesh2 = get_mixture_mesh(file1, file2)

        mixname1 = f'{u}_{v}.obj'
        mixname2 = f'{v}_{u}.obj'
        mesh1.export(os.path.join('mixture',mixname1))
        mesh2.export(os.path.join('mixture',mixname2))

        # break






