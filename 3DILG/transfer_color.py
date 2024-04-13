import trimesh
import numpy as np
import os
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=45572)
args = parser.parse_args()

point_fold = os.path.join('/public/home/zhaoyq', '3D-FUTURE-model_sdf')

pc_folder = os.path.join('/public/home/zhaoyq', '3D-FUTURE-points')
surf_folder = os.path.join('/public/home/zhaoyq', '3D-FUTURE-model_watertight')

to_path = os.path.join('/public/home/zhaoyq', '3D-FUTURE-model-points1')

st = []
with open(os.path.join(point_fold, 'eval.csv')) as f:
    for u in f:
        st.append(u.strip())

# st = list(st)
ps_len = len(st)
print(ps_len, args.start, args.end)
for i in tqdm(range(args.start, args.end)):
    npy_path = os.path.join(pc_folder, f'{st[i]}_points.npy')
    mesh = trimesh.load(os.path.join(surf_folder, f'{st[i]}.obj'))
    surface = np.load(npy_path)
    surface[..., 1] = -surface[..., 1]
    surface[..., 2] = -surface[..., 2]
    for sj in range(10):
        surface_w = surface[sj * 100000:(sj + 1) * 100000, :3]
        if surface_w.shape[0] == 0:
            continue
        changed, _, _ = trimesh.proximity.closest_point(mesh, surface_w)
        surface[sj * 100000:(sj + 1) * 100000, :3] = changed

    np.save(os.path.join(to_path, f'{st[i]}_points.npy'), surface)

