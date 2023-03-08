import numpy as np
import torch
import cv2
import os
import json
import math as m
import matplotlib.pyplot as plt

from functools import cached_property

from loaders.camera_geometry_loader import IndexMapping


def build_intrinsics(fx, fy, cx, cy):
    K = torch.eye(3)
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
    return torch.Tensor(new_pose)


class SyntheticLoader(object):
    def __init__(self,
        scan_path:str,
        frame_range:tuple[int, int],
        frame_stride:int,
        image_scale:float,
        load_images_bool:bool=True):

        self.scan_path = scan_path
        self.frame_range = frame_range
        self.frame_stride = frame_stride
        self.image_scale = image_scale
        self.load_images_bool = load_images_bool
        self.load_depths_bool = False

        with open(os.path.join(self.scan_path, 'transforms_train.json'), 'r') as fp:
            transforms = json.load(fp)

        # # formating framerange
        full_length = len(transforms['frames'])

        if self.frame_range is None:
            self.frame_range = (0, full_length)
        else:
            if self.frame_range[0] is None:
                self.frame_range[0] = 0
            elif self.frame_range[0] < 0:
                self.frame_range[0] = full_length - self.frame_range[0]

            if self.frame_range[1] is None:
                self.frame_range[1] = full_length
            elif self.frame_range[1] < 0:
                self.frame_range[1] = full_length - self.frame_range[1]

        if self.frame_stride is None:
            self.frame_stride = 1

        # for i in range(full_length):

        images = []
        intrinsics = []
        extrinsics = []

        for i in range(self.frame_range[0], self.frame_range[1], self.frame_stride):
            if self.load_images_bool:
                image_path = os.path.join(self.scan_path, transforms['frames'][i]['file_path']) + '.png'
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = torch.ByteTensor(image)
                images.append(image)

            # Get calibration
            H, W = image.shape[:2]
            alpha = float(transforms['camera_angle_x'])
            focal = 0.5 * W / np.tan(0.5 * alpha)
            intrinsics.append(build_intrinsics(focal, focal, W/2, H/2))

            extrinsics.append(nerf_matrix_to_ngp(np.array(transforms['frames'][i]['transform_matrix']), 1))

        self.index_mapping = IndexMapping([[1, len(intrinsics)]])

        if self.load_images_bool:
            self.images = torch.stack(images, dim=0)
        self.intrinsics = torch.stack(intrinsics, dim=0)
        self.extrinsics = torch.stack(extrinsics, dim=0)

        self.N, self.H, self.W = self.images.shape[:3]

    def format_groundtruth(self, gt, background=None):
        # If image data is stored as uint8, convert to float32 and scale to (0, 1)
        # is alpha channel is present, add random background color to image data
        if background is None:
            color_bg = torch.rand(3, device=gt.device) # [3], frame-wise random.
        else:
            color_bg = torch.Tensor(background).to(gt.device)

        if gt.shape[-1] == 4:
            if self.images.dtype == torch.uint8:
                rgba_gt = (gt.to(torch.float32) / 255).to(gt.device)
            else:
                rgba_gt = gt.to(gt.device)
            rgb_gt = rgba_gt[..., :3] * rgba_gt[..., 3:] + color_bg * (1 - rgba_gt[..., 3:])
        else:
            if self.images.dtype == torch.uint8:
                rgb_gt = (gt.to(torch.float32) / 255).to(gt.device)
            else:
                rgb_gt = gt.to(gt.device)

        return rgb_gt, color_bg

    
    def get_custom_batch(self, n, h, w, background, device):
        K = self.intrinsics[n].to(device)
        E = self.extrinsics[n].to(device)

        if self.load_images_bool:
            rgb_gt, color_bg = self.format_groundtruth(self.images[n, h, w, :].to(device), background)
        else:
            rgb_gt, color_bg = None, None
        if self.load_depths_bool:
            depth = self.depths[n, h, w].to(device)
        else:
            depth = None

        n = n.to(device)
        h = h.to(device)
        w = w.to(device)

        return n, h, w, K, E, rgb_gt, color_bg, depth


    def get_random_batch(self, batch_size, device):
        n = torch.randint(0, self.N, (batch_size,))
        h = torch.randint(0, self.H, (batch_size,))
        w = torch.randint(0, self.W, (batch_size,))

        return self.get_custom_batch(n, h, w, background=None, device=device)


    def get_image_batch(self, image_num, device):
        h = torch.arange(0, self.H)
        w = torch.arange(0, self.W)
        h, w = torch.meshgrid(h, w, indexing='ij')
        n = torch.full(h.shape, fill_value=image_num)

        return self.get_custom_batch(n, h, w, background=(1, 1, 1), device=device)


    def get_pointcloud_batch(self, cams, freq, side_margin, device):

        for i in range(self.N):
            n = []
            s, r, c = self.index_mapping.idx_to_src(i)
            if r > side_margin and r < self.index_mapping.get_num_rigs(s) - side_margin:
                if c in cams:
                    if r % freq == 0:
                        n.append(i)

                n = torch.Tensor(np.array(n)).to(int)
                h = torch.arange(0, self.H)
                w = torch.arange(0, self.W)
                n, h, w = torch.meshgrid(n, h, w, indexing='ij')

                yield self.get_custom_batch(n, h, w, background=(1, 1, 1), device=device)


    def get_calibration(self, device):
        K = self.intrinsics.to(device)
        E = self.extrinsics.to(device)

        return self.N, self.H, self.W, K, E
    
    @cached_property
    def translation_center(self):
        return torch.mean(self.extrinsics[..., :3, 3], dim=0, keepdims=True)