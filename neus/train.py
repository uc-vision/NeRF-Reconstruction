import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader import CameraGeometryLoader
from loaders.synthetic import SyntheticLoader

from nerf.nets import NeRFCoordinateWrapper, NeRFNetwork, Transform, NeRF
from nerf.trainer import NeRFTrainer
from nerf.logger import Logger
from nerf.metrics import PSNRWrapper, SSIMWrapper, LPIPSWrapper
from nerf.render import Render
# from nerf.inference import ImageInference, InvdepthThreshInference, PointcloudInference

# from nets import NeRFCoordinateWrapper, NeRFNetwork
from neus.trainer import NeusNeRFTrainer
from neus.render import NeuSRenderer
from neus.inference import ImageInference
from neus.nets import SDFNetwork, RenderingNetwork, SingleVarianceNetwork

# from misc import configurator


def remove_backgrounds(dataloader, max_depth):
    mask = torch.full(dataloader.depths.shape, fill_value=255, dtype=torch.uint8)[..., None]
    mask[dataloader.depths > 1] = 0
    dataloader.images = torch.cat([dataloader.images, mask], dim=-1)


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:

    cfg_nerf = OmegaConf.load("./nerf/logs/plant_and_food/test3/20230302_152237/config.yaml")

    logger = Logger(
        root_dir=cfg.log.root_dir,
        cfg=cfg,
        )

    ### Load OG NeRF
    if cfg.nerf_init.path is not None:
        logger.log('Initiating Dataloader...')
        if cfg_nerf.scan.loader == "camera_geometry":
            dataloader_nerf = CameraGeometryLoader(
                scan_paths=cfg_nerf.scan.scan_paths,
                scan_pose_paths=cfg_nerf.scan.scan_pose_paths,
                frame_ranges=cfg_nerf.scan.frame_ranges,
                frame_strides=cfg_nerf.scan.frame_strides,
                image_scale=cfg_nerf.scan.image_scale,
                load_images_bool=False,
                )
        elif cfg.nerf.scan.loader == "synthetic":
            dataloader_nerf = SyntheticLoader(
                scan_path=cfg_nerf.scan.scan_paths,
                frame_range=cfg_nerf.scan.frame_ranges,
                frame_stride=cfg_nerf.scan.frame_strides,
                image_scale=cfg_nerf.scan.image_scale,
                load_images_bool=False,
                )

        
        logger.log('Initilising Model...')
        model_nerf = NeRFNetwork(
            N = dataloader_nerf.N,
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
        model_nerf.load_state_dict(torch.load(cfg.nerf_init.path))

        transform = Transform(translation=-dataloader_nerf.translation_center()).to('cuda')

        model_nerf_coord = NeRFCoordinateWrapper(
            model=model_nerf,
            transform=transform,
            inner_bound=cfg.scan.inner_bound,
            outer_bound=cfg.scan.outer_bound,
        ).to('cuda')

        renderer_nerf = Render(
            models=model_nerf_coord,
            steps_firstpass=cfg_nerf.renderer_thresh.steps,
            z_bounds=cfg_nerf.renderer_thresh.z_bounds,
            steps_importance=cfg_nerf.renderer_thresh.importance_steps,
            alpha_importance=cfg_nerf.renderer_thresh.alpha,
        )
    else:
        renderer_nerf = None

    ### Init SDF NeRF
    if cfg.scan.loader == "camera_geometry":
        dataloader = CameraGeometryLoader(
            scan_paths=cfg.scan.scan_paths,
            scan_pose_paths=cfg.scan.scan_pose_paths,
            frame_ranges=cfg.scan.frame_ranges,
            frame_strides=cfg.scan.frame_strides,
            image_scale=cfg.scan.image_scale,
            )
    elif cfg.scan.loader == "synthetic":
        dataloader = SyntheticLoader(
            scan_path=cfg.scan.scan_paths[0],
            frame_range=cfg.scan.frame_ranges[0],
            frame_stride=cfg.scan.frame_strides[0],
            image_scale=cfg.scan.image_scale,
            )
        
    sdf_network = SDFNetwork().to("cuda")
    color_network = RenderingNetwork().to("cuda")
    deviation_network = SingleVarianceNetwork(0.6).to("cuda")


    renderer_sdf = NeuSRenderer(
        sdf_network=sdf_network,
        color_network=color_network,
        deviation_network=deviation_network,
        renderer_nerf=renderer_nerf,
        steps_firstpass=cfg.renderer.steps,
        z_bounds=cfg.renderer.z_bounds,
        )

    if cfg_nerf.inference.image.image_num == 'middle':
        rigs_num = dataloader.index_mapping.get_num_rigs()
        inference_image_num = int(dataloader.index_mapping.src_to_idx(0, rigs_num // 2, 3).item())
    else:
        inference_image_num = cfg_nerf.inference.image.image_num

    logger.log('Initiating Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'sdf_encoding', 'params': list(renderer_sdf.sdf_network.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'sdf_net', 'params': list(renderer_sdf.sdf_network.network.parameters()), 'lr': cfg.optimizer.net.lr},
            {'name': 'color', 'params': list(renderer_sdf.color_network.parameters()), 'lr': cfg.optimizer.net.lr},
            {'name': 'deviation_network', 'params': list(renderer_sdf.deviation_network.parameters()), 'lr': 1e-3},
        ], betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

    if cfg.scheduler == 'step':
        lmbda = lambda x: 1
    elif cfg.scheduler == 'exp_decay':
        lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
    elif cfg.scheduler == 'warmup':
        # alpha = 0.05
        # learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        # lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
        def lmbda(iter_step):
            warm_up_end = 500
            end_iter = 300000
            if iter_step < warm_up_end:
                learning_factor = iter_step / warm_up_end
            else:
                alpha = 0.05
                progress = (iter_step - warm_up_end) / (end_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
            return learning_factor
    else:
        raise ValueError
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    inferencers_sdf = {
        "image": ImageInference(
            renderer_sdf, dataloader, cfg.trainer.n_rays, inference_image_num, optimizer),
    }
    
    logger.log('Initiating Trainer...')
    trainer = NeusNeRFTrainer(
        dataloader=dataloader,
        logger=logger,
        renderer=renderer_sdf,
        inferencers=inferencers_sdf,

        optimizer=optimizer,
        scheduler=scheduler, 
        
        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,

        # dist_loss_range=cfg.trainer.dist_loss_range,
        # depth_loss_range=cfg.trainer.depth_loss_range,

        eval_image_freq=cfg.log.eval_image_freq,
        eval_pointcloud_freq=cfg.log.eval_pointcloud_freq,
        save_weights_freq=cfg.log.save_weights_freq,

        metrics={},
        )

    logger.log('Beginning Training...\n')
    trainer.train()


if __name__ == '__main__':
    train()