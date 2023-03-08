import numpy as np
import torch
import cv2
import os
import json
import math as m
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)


def build_intrinsics(fx, fy, cx, cy):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    return K


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def load_image_set(rootdir, near=2, far=6, scale=1):
    with open(os.path.join(rootdir, 'transforms_train.json')) as fp:
        transforms = json.load(fp)

    images = []
    intrinsics = []
    extrinsics = []
    bds = []

    for i in range(len(transforms['frames'])):
        image_path = os.path.join(rootdir, transforms['frames'][i]['file_path']) + '.png'

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        image = image.astype(np.float32) / 255

        images.append(image)

        # Get calibration
        H, W = image.shape[:2]
        alpha = float(transforms['camera_angle_x'])
        focal = 0.5 * W / np.tan(0.5 * alpha)
        intrinsics.append(build_intrinsics(focal, focal, W/2, H/2))

        # extrinsics.append(np.array(transforms['frames'][i]['transform_matrix']))
        extrinsics.append(nerf_matrix_to_ngp(np.array(transforms['frames'][i]['transform_matrix']), 0.8))

        bds.append(np.array([near, far], dtype=np.float32))

    images = np.stack(images, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)
    extrinsics = np.stack(extrinsics, axis=0)
    bds = np.stack(bds, axis=0)

    N, H, W, _ = np.shape(images)

    depths = np.ones((N, H, W))

    ## scaling to fit within (-1, 1)
    # scale = 100
    # extrinsics[:,:3,3] = extrinsics[:,:3,3] * scale
    # extrinsics[:,:3,3] = extrinsics[:,:3,3] + 0.5
    # bds = bds * scale

    # extrinsics = nerf_matrix_to_ngp(extrinsics)

    #
    # extrinsics[:,:3,1] *= -1
    # extrinsics[:,:3,2] *= -1

    # print(np.amax(extrinsics[:,:3,3]), np.amin(extrinsics[:,:3,3]))

    return images, depths, intrinsics, extrinsics, bds