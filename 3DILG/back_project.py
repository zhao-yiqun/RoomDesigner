import numpy as np
import json
import os
import cv2
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm
import argparse


# proj =  np.array([
#         [[ 2.7778,  0.0000,  0.0000,  0.0000],
#          [ 0.0000, -2.7778,  0.0000,  0.0000],
#          [ 0.0000,  0.0000, -1.0002, -0.2000],
#          [ 0.0000,  0.0000, -1.0000,  0.0000]]])
#
#
# def normalize_vecs(vectors):
#     return vectors/np.linalg.norm(vectors)
#
#
# def create_my_world2cam_matrix(forward_vector, origin):
#     forward_vector = normalize_vecs(forward_vector)
#     up_vector = np.array([0, 1, 0])
#     left_vector = normalize_vecs(np.cross(up_vector, forward_vector))
#     up_vector=  normalize_vecs(np.cross(forward_vector, left_vector))
#     new_t = np.eye(4)
#     new_t[:3, 3] = -origin
#     new_r = np.eye(4)
#     new_r[:3, :3] = np.concatenate((left_vector[None], up_vector[None], forward_vector[None]), axis=0)
#     world2cam = new_r @ new_t
#     return world2cam

#
# def back_project_point_cloud(path, filename):
#     pointclouds = []
#     for i in range(5):
#         filed = "%05d" % i
#         json_path = os.path.join(path, filename, f'{filed}.json')
#         world_2_cam_matrix = get_trans(json_path)
#         depth_img = os.path.join(path, filename, f'{filed}_depth.png')
#         depth = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
#         scale = 5 / 65535
#         depth = depth * scale
#         depth = -depth
#         mask_img = os.path.join(path, filename, f'{filed}_a.png')
#         mask = cv2.imread(mask_img, cv2.IMREAD_UNCHANGED)
#         A, B = -1.0002, -0.2000
#         ndc = -A - B / depth
#         x = (np.linspace(511, -512, 1024) + 0.5) / 512
#         y = (np.linspace(511, -512, 1024) + 0.5) / 512
#         xv, yv = np.meshgrid(x, y, indexing='ij')
#         coords_ndc = -depth[..., None] * np.concatenate(
#             (yv[..., None], -xv[..., None], ndc[..., None], np.ones_like(ndc[..., None])), axis=2)
#         coords_c = np.dot(coords_ndc, np.linalg.inv(proj[0]).transpose(1, 0))
#         coords_w = np.dot(coords_c, np.linalg.inv(world_2_cam_matrix).transpose(1, 0))
#         points = coords_w[(mask > 0) & (depth > -5)]
#         r = cv2.imread(os.path.join(path, filename, f'{filed}_r.png')
#                               , cv2.IMREAD_UNCHANGED)
#         r = r[(mask > 0) & (depth > -5)]//256
#         g = cv2.imread(os.path.join(path, filename, f'{filed}_g.png')
#                        , cv2.IMREAD_UNCHANGED)
#         g = g[(mask > 0) & (depth > -5)]//256
#         b = cv2.imread(os.path.join(path, filename, f'{filed}_b.png')
#                        , cv2.IMREAD_UNCHANGED)
#         b = b[(mask > 0) & (depth > -5)]//256
#         x, y, z = -points[:, 2], points[:, 1], -points[:, 0]
#         rgbpoints = np.concatenate((x[..., None], y[..., None], z[..., None], np.ones_like(z[..., None])* i)
#                                    , axis=1)
#         # rgbpoints = np.concatenate((x[..., None], y[..., None], z[..., None], r[..., None], g[..., None], b[..., None])
#         #                            ,axis=1)
#         pointclouds.append(rgbpoints)
#
#     pointclouds = np.concatenate((pointclouds), axis=0)
#
#     return pointclouds

def depth_image_to_point_cloud(rgb, depth, scale, K, pose, mask):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)
    mask = np.ravel(mask)
    valid = (Z > 0) & (mask > 0)

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(np.linalg.inv(pose), position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B)))

    return points


def get_trans(path):
    with open(path) as f:
        file = json.load(f)
    return file

def get_camera(filed, json_f):
    # print(json_f)
    files = json_f[filed]
    projectionMatrix = np.matrix(files['nP'])
    intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(projectionMatrix)[:3]

    camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
    camR = Rot.from_matrix(rotationMatrix)
    tvec = camR.apply(camT.ravel())
    RT_obj = np.array(
        [[1.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000],
         [0.0000, -1.0000, 0.0000]]
    )

    return tvec, camR, RT_obj, intrinsic


def back_project_point_cloud(path, filename):
    pointclouds = []
    json_path = os.path.join(path, filename, 'info.json')
    json_f = get_trans(json_path)
    for i in range(20):
        filed = "%05d" % i
        tvec, camR, RT_obj, intrinsic = get_camera(f'{filed}frame', json_f['frames'][i])
        depth_img = os.path.join(path, filename, f'{filed}_depth.png')
        depth = cv2.imread(depth_img, cv2.IMREAD_UNCHANGED)
        scale = 6.5 / 65535
        depth = depth * scale
        depth[depth==6.5] = 0
        mask_img = os.path.join(path, filename, f'{filed}_a.png')
        mask = cv2.imread(mask_img, cv2.IMREAD_UNCHANGED)
        r = cv2.imread(os.path.join(path, filename, f'{filed}_r.png')
                              , cv2.IMREAD_UNCHANGED)
        r = r/65535
        g = cv2.imread(os.path.join(path, filename, f'{filed}_g.png')
                       , cv2.IMREAD_UNCHANGED)
        g = g/65535
        b = cv2.imread(os.path.join(path, filename, f'{filed}_b.png')
                       , cv2.IMREAD_UNCHANGED)
        b = b/65535

        rgb = np.concatenate((r[..., None], g[..., None], b[..., None])
                                   , axis=2)
        world2cam = np.eye(4)
        world2cam[:3, :3] = camR.as_matrix()
        world2cam[:3, 3] = tvec

        rgbpoints =depth_image_to_point_cloud(rgb, depth, 1, intrinsic, world2cam, mask)
        rgbpoints[..., :3] = np.dot(rgbpoints[..., :3], RT_obj)
        pointclouds.append(rgbpoints)

    pointclouds = np.concatenate((pointclouds), axis=0)

    return pointclouds



def main():
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')

    parser.add_argument(
        '--begin', type=int, default=0,
        help='begining ids'
    )
    parser.add_argument(
        '--end', type=int, default=2000,
        help='end ids'
    )
    args = parser.parse_args()
    path = '/storage/data/zhaoyq/3D-FUTURE-renders'
    # filename = '020aa1aa-426d-4b52-85cf-fe05cc0b53ab'
    # filename = 'output'
    to_path = '/storage/data/zhaoyq/3D-FUTURE-points'
    with open('/storage/data/zhaoyq/3D-FUTURE-renders/transd2.csv') as f:
        rows = [row.strip() for row in f]
    for i in  tqdm(range(args.begin, args.end)):
        filename = rows[i]
        point_cloud = back_project_point_cloud(path, filename)
        # print(point_cloud.shape)
        if point_cloud.shape[0] < 1000000:
            ind = np.array(range(point_cloud.shape[0]))
        else:
            ind = np.random.default_rng().choice(point_cloud.shape[0], 1000000, replace=False)
        # print(point_cloud[..., :3].max(), point_cloud[..., :3].min())
        np.save(os.path.join(to_path, f'{filename}_points.npy'), point_cloud[ind])
    # np.save(os.path.join('points.npy'), point_cloud[ind])


if __name__ == '__main__':
    main()