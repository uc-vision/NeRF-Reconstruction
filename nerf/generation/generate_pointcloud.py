import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
import open3d as o3d

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader import CameraGeometryLoader

from nerf.nets import NeRFCoordinateWrapper, NeRFNetwork, Transform, NeRF
from nerf.trainer import NeRFTrainer
from nerf.logger import Logger
from nerf.metrics import PSNRWrapper, SSIMWrapper, LPIPSWrapper
from nerf.render import Render
from nerf.inference import ImageInference, InvdepthThreshInference, PointcloudInference


def convert_pointcloud(pointcloud_npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_npy['points'])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud_npy['colors'])
    return pcd


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def generate_pointcloud(cfg : DictConfig) -> None:


    logger = Logger(
        root_dir=cfg.log.root_dir,
        cfg=cfg,
        )

    cfg_nerf = OmegaConf.load("./nerf/logs/plant_and_food/test3/20230302_152237/config.yaml")

    # logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scan_paths=cfg_nerf.scan.scan_paths,
        scan_pose_paths=cfg_nerf.scan.scan_pose_paths,
        frame_ranges=cfg_nerf.scan.frame_ranges,
        frame_strides=cfg_nerf.scan.frame_strides,
        image_scale=cfg_nerf.scan.image_scale,
        load_images_bool=False,
        )
    
    # logger.log('Initilising Model...')
    model = NeRFNetwork(
        N = dataloader.extrinsics.shape[0],
        encoding_precision=cfg_nerf.nets.encoding.precision,
        encoding_n_levels=cfg_nerf.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg_nerf.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg_nerf.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg_nerf.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg_nerf.nets.sigma.hidden_dim,
        sigma_num_layers=cfg_nerf.nets.sigma.num_layers,
        encoding_dir_precision=cfg_nerf.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg_nerf.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg_nerf.nets.encoding_dir.degree,
        latent_embedding_dim=cfg_nerf.nets.latent_embedding.features,
        color_hidden_dim=cfg_nerf.nets.color.hidden_dim,
        color_num_layers=cfg_nerf.nets.color.num_layers,
    ).to('cuda')
    # model.load_state_dict(torch.load("./nerf/logs/plant_and_food/test2/20230220_194316/model/10000.pth"))
    model.load_state_dict(torch.load("./nerf/logs/plant_and_food/test3/20230302_152237/model/10000.pth"))

    transform = Transform(translation=-dataloader.translation_center).to('cuda')

    model_coord = NeRFCoordinateWrapper(
        model=model,
        transform=transform,
        # transform=None,
        inner_bound=cfg_nerf.scan.inner_bound,
        outer_bound=cfg_nerf.scan.outer_bound,
    ).to('cuda')

    renderer = Render(
        models=model_coord,
        steps_firstpass=cfg_nerf.renderer.steps,
        z_bounds=cfg_nerf.renderer.z_bounds,
        steps_importance=cfg_nerf.renderer.importance_steps,
        alpha_importance=cfg_nerf.renderer.alpha,
    )

    renderer_thresh = Render(
        models=model_coord,
        steps_firstpass=cfg_nerf.renderer_thresh.steps,
        z_bounds=cfg_nerf.renderer_thresh.z_bounds,
        steps_importance=cfg_nerf.renderer_thresh.importance_steps,
        alpha_importance=cfg_nerf.renderer_thresh.alpha,
    )

    if cfg_nerf.inference.image.image_num == 'middle':
        rigs_num = dataloader.index_mapping.get_num_rigs()
        inference_image_num = int(dataloader.index_mapping.src_to_idx(0, rigs_num // 2, 3).item())
    else:
        inference_image_num = cfg_nerf.inference.image.image_num

    inferencers = {
        "image": ImageInference(
            renderer, dataloader, cfg_nerf.trainer.n_rays, inference_image_num),
        "invdepth_thresh": InvdepthThreshInference(
            renderer_thresh, dataloader, cfg_nerf.trainer.n_rays, inference_image_num),
        "pointcloud": PointcloudInference(
            renderer_thresh,
            dataloader,
            cfg_nerf.inference.pointcloud.max_variance,
            cfg_nerf.inference.pointcloud.distribution_area,
            cfg_nerf.trainer.n_rays,
            cfg_nerf.inference.pointcloud.cams,
            cfg_nerf.inference.pointcloud.freq * 10,
            cfg_nerf.inference.pointcloud.side_margin)
    }

    pointcloud = inferencers['pointcloud']()
    o3d.io.write_point_cloud("./nerf/logs/plant_and_food/test3/20230302_152237/pointclouds/10000.pcd", convert_pointcloud(pointcloud))

if __name__ == '__main__':
    generate_pointcloud()