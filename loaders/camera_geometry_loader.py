import numpy as np
import torch
import gc

from tqdm import tqdm
from pathlib import Path
from functools import cached_property

from camera_geometry.scan.scan import Scan
from camera_geometry.json import load_json
from camera_geometry.scan.views import load_frames
from camera_geometry.concurrent import expand, map_lists


def load_calibrations(scan_list, scan_pose_list):
    intrinsics = []
    extrinsics = []
    for scan, scan_pose in zip(scan_list, scan_pose_list):
        for rig_pose in scan.rig_poses:
            for key, camera in scan.cameras.items():
                camera_undistored = camera.optimal_undistorted().undistorted
                intrinsics.append(camera_undistored.intrinsic)
                extrinsics.append(scan_pose @ rig_pose @ camera_undistored.parent_to_camera)
    intrinsics = torch.Tensor(np.stack(intrinsics, axis=0))
    extrinsics = torch.Tensor(np.stack(extrinsics, axis=0))
    return intrinsics, extrinsics


def load_images(scan_list):

    def load_image(undist, image_file):
        rgb = scan.loader.rgb.load_image(image_file)
        rgb_undist = undist.undistort(rgb)
        del undist.undistort_map
        return torch.ByteTensor(rgb_undist)

    params = []

    for scan in scan_list:
        for rig_index, rig_pose in enumerate(scan.rig_poses):
            for key, camera in scan.cameras.items():
                image_file = scan.image_sets.rgb[rig_index][key]
                params.append([(camera.optimal_undistorted(), image_file)])

    images = map_lists(expand(load_image), params)
    for i in range(len(images)):
        images[i] = images[i][0]
    images_tensor = torch.stack(images, dim=0)
    del images
    gc.collect()

    return images_tensor


def load_depths(scan_list):

    def load_image(undist, image_file):
        depth = undist.undistort(scan.loader.depth.load_image(image_file)) * 16 / 65536
        return torch.Tensor(depth)

    params = []

    for scan in scan_list:
        for rig_index, rig_pose in enumerate(scan.rig_poses):
            for key, camera in scan.cameras.items():
                depth_file = scan.image_sets.depth[rig_index][key]
                params.append([(camera.optimal_undistorted(), depth_file)])

    depths = map_lists(expand(load_image), params)
    for i in range(len(depths)):
        depths[i] = depths[i][0]

    return torch.stack(depths, dim=0)



class IndexMapping(object):
    def __init__(self, scan_dims:list[tuple[int, int]]):
        self.scan_dims = torch.Tensor(scan_dims)
        self.scan_counts = torch.prod(self.scan_dims, dim=1)
        self.scan_start_index = torch.cumsum(self.scan_counts, dim=0) - self.scan_counts
        self.scan_end_index = torch.cumsum(self.scan_counts, dim=0)

    def idx_to_src(self, idx):
        # assert (idx < self.scan_end_index[-1]).all() & (idx >= 0).all(), "index outside range"
        scan = torch.searchsorted(self.scan_start_index, idx, right=True) - 1
        idx_in_scan = idx - self.scan_start_index[scan]
        rig_in_scan = torch.div(idx_in_scan, self.scan_dims[scan, 1], rounding_mode='floor')
        cam_in_rig = torch.remainder(idx_in_scan, self.scan_dims[scan, 1])
        return scan, rig_in_scan, cam_in_rig

    def idx_to_rig(self, idx):
        scan = torch.searchsorted(self.scan_start_index, idx, right=True) - 1
        idx_in_scan = idx - self.scan_start_index[scan]
        rig_in_scan = torch.div(idx_in_scan, self.scan_dims[scan, 1], rounding_mode='floor')
        rig = rig_in_scan + torch.cumsum(self.scan_dims, dim=0)[scan, 0] - self.scan_dims[0]
        return rig


    def src_to_idx(self, scan, rig, camera):
        idx = self.scan_start_index[scan] + rig * self.scan_dims[scan, 1] + camera
        return idx

    def get_num_scans(self):
        return self.scan_dims.shape[0]

    def get_num_rigs(self, scan=None):
        if scan is None:
            return torch.sum(self.scan_dims, dim=0)[0]
        else:
            return self.scan_dims[scan, 0]

    def get_num_cams(self, scan=None, rig:bool=False):
        if scan is None:
            return torch.sum(self.scan_counts)
        elif rig:
            return self.scan_dims[scan, 1]
        else:
            return self.scan_counts[scan]


class CameraGeometryLoader(object):

    def __init__(self,
        scan_paths:list[str],
        scan_pose_paths:list[str],
        frame_ranges:list[tuple[int, int]],
        frame_strides:list[int],
        image_scale:float,
        load_images_bool:bool=True,
        load_depths_bool:bool=False):

        scan_list = []
        scan_pose_list = []

        self.load_images_bool = load_images_bool
        self.load_depths_bool = load_depths_bool

        scan_dims = []

        for scan_path, frame_range, frame_stride, scan_pose_path in zip(scan_paths, frame_ranges, frame_strides, scan_pose_paths):

            scan = Scan.load(scan_path)
            scan = scan.with_image_scale(image_scale)
            scan = scan.with_frames(frame_range=frame_range, frame_stride=frame_stride)

            if scan_pose_path is not None:
                scan_pose = np.load(scan_pose_path)
            else:
                scan_pose = np.eye(4)

            scan_list.append(scan)
            scan_pose_list.append(scan_pose)

            scan_dims.append((len(scan.rig_poses), len(scan.cameras)))

        self.index_mapping = IndexMapping(scan_dims)

        self.intrinsics, self.extrinsics = load_calibrations(scan_list, scan_pose_list)

        # self.N = self.intrinsics.shape[0]
        # self.H = scan_list[0].common_image_size()[1]
        # self.W = scan_list[0].common_image_size()[0]

        # self.translation_center = torch.mean(self.extrinsics[..., :3, 3], dim=0, keepdims=True)

        if self.load_images_bool:
            self.images = load_images(scan_list)
        # else:
        #     self.images = torch.zeros((self.N, self.H, self.W, 3), dtype=torch.uint8)
        #     self.load_images_bool = True

        if self.load_depths_bool:
            self.depths = load_depths(scan_list)


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


# def index_mapping_test():
#     # scan_dims = torch.Tensor(np.array([[10, 8], [12, 6], [14, 6]], dtype=int))
#     scan_dims = torch.LongTensor([[10, 8], [12, 6], [14, 6]])
#     # print(scan_dims.shape)
#     index_mapping = IndexMapping(scan_dims=scan_dims)
#     # index_map = IndexMapping()

#     # print(index_mapping.scan_starting_index)

#     print(index_mapping.idx_to_src(0))
#     print(index_mapping.idx_to_src(7))
#     print(index_mapping.idx_to_src(8))
#     print(index_mapping.idx_to_src(79))
#     print(index_mapping.idx_to_src(80))
#     print(index_mapping.idx_to_src(10*8+12*6))
#     print(index_mapping.idx_to_src(10*8+12*6+14*6-1))
    
#     idxs = torch.LongTensor([8, 9, 11, 94, 183, -1])
#     print(index_mapping.idx_to_src(idxs))

#     print(index_mapping.src_to_idx(2, 3, 4))
#     print(index_mapping.idx_to_src(index_mapping.src_to_idx(0, 0, 0)))
#     print(index_mapping.idx_to_src(index_mapping.src_to_idx(2, 3, 4)))
#     # print(index_mapping.idx_to_src(index_mapping.src_to_idx(2, 3, 4)))



# if __name__ == '__main__':
#     # scan_paths = [
#     #     '/mnt/maara/conan_scans/blenheim-21-6-28/30-06_13-05/ROW_349_EAST_SLOW_0006/scene.json',
#     #     '/mnt/maara/conan_scans/blenheim-21-6-28/30-06_13-05/ROW_349_WEST_SLOW_0007/scene.json']

#     # data_loader = CameraGeometryLoader(
#     #     scan_paths=scan_paths, 
#     #     scan_pose_paths=[None, None], 
#     #     frame_ranges=[(0, 20), (0, 30)], 
#     #     frame_strides=[None, None], 
#     #     image_scale=0.5)

#     scan_paths = [
#         "remote://row/178/50?span=1&row_wise=400"]
#         # "remote://frame_opposite/39079?span=2"]

#     data_loader = CameraGeometryLoader(
#         scan_paths=scan_paths, 
#         scan_pose_paths=[None], 
#         frame_ranges=[None], 
#         frame_strides=[None], 
#         image_scale=0.5)

#     # index_mapping_test()

#     # n, h, w, K, E, rgb_gt, color_bg = data_loader.get_random_batch(32)
#     # n, h, w, K, E, rgb_gt, color_bg = data_loader.get_image_batch(32)

#     # print(n.shape, h.shape, K.shape, E.shape, rgb_gt.shape, color_bg)