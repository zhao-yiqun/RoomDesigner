import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation as Rot
from matplotlib import pyplot as plt
import json

with open('/home/yiqun/Scenesynthesis/ATISS/GET3D/render_shapenet_data/output/00001.json') as f:
    file = json.load(f)


projectionMatrix = np.matrix(file['nP'])
intrinsic, rotationMatrix, homogeneousTranslationVector = cv.decomposeProjectionMatrix(projectionMatrix)[:3]

camT = -cv.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
camR = Rot.from_matrix(rotationMatrix)
tvec = camR.apply(camT.ravel())
rvec = camR.as_rotvec()

objectPoint = np.load('/home/yiqun/Scenesynthesis/ATISS/GET3D/render_shapenet_data/point_cloud.npy')
RT_obj = np.array(
    [[1.0000, 0.0000, 0.0000],
    [0.0000, 0.0000, 1.0000],
    [0.0000, -1.0000, 0.0000]]
)
objectPoint = np.dot(objectPoint, RT_obj)
# change to opencv's world space
# objectPoint = np.load(file['nT'])

projectedObjectPoint = cv.projectPoints(
    objectPoints=objectPoint,
    rvec=rvec,
    tvec=tvec,
    cameraMatrix=intrinsic,
    distCoeffs=None,
)

print(intrinsic, rvec, tvec)

img = np.zeros((1024, 1024, 3), np.uint8)

print(projectedObjectPoint)
projectedObjectPoint = np.round(projectedObjectPoint[0].squeeze()).astype(int)



for i in range(200000):
    img = cv.circle(
        img,
        center=projectedObjectPoint[i],
        radius=0,
        color=(255, 0, 0),
        thickness=2
    )

plt.imshow(img)
plt.show()

import ipdb
ipdb.set_trace()