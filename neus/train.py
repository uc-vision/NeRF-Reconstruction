import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import DictConfig, OmegaConf

from loaders.camera_geometry_loader import CameraGeometryLoader

from nerf.nets import NeRFCoordinateWrapper, NeRFNetwork, Transform, NeRF
from nerf.trainer import NeRFTrainer
from nerf.logger import Logger
from nerf.metrics import PSNRWrapper, SSIMWrapper, LPIPSWrapper
from nerf.render import Render
# from nerf.inference import ImageInference, InvdepthThreshInference, PointcloudInference

# from nets import NeRFCoordinateWrapper, NeRFNetwork
from neus.trainer import NeusNeRFTrainer
from neus.render2 import NeuSRenderer
from neus.inference import ImageInference

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
    logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scan_paths=cfg_nerf.scan.scan_paths,
        scan_pose_paths=cfg_nerf.scan.scan_pose_paths,
        frame_ranges=cfg_nerf.scan.frame_ranges,
        frame_strides=cfg_nerf.scan.frame_strides,
        image_scale=cfg_nerf.scan.image_scale,
        # load_images_bool=False,
        )

    
    logger.log('Initilising Model...')
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
    model.load_state_dict(torch.load("./nerf/logs/plant_and_food/test3/20230302_152237/model/10000.pth"))

    transform = Transform(translation=-dataloader.translation_center).to('cuda')

    model_coord = NeRFCoordinateWrapper(
        model=model,
        transform=transform,
        inner_bound=cfg.scan.inner_bound,
        outer_bound=cfg.scan.outer_bound,
    ).to('cuda')

    metrics = {
        "eval_lpips": LPIPSWrapper(),
        "eval_ssim": SSIMWrapper(),
        "eval_psnr": PSNRWrapper(),
    }

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

    # inferencers = {
    #     "image": ImageInference(
    #         renderer, dataloader, cfg_nerf.trainer.n_rays, inference_image_num),
    #     "invdepth_thresh": InvdepthThreshInference(
    #         renderer_thresh, dataloader, cfg_nerf.trainer.n_rays, inference_image_num),
    #     "pointcloud": PointcloudInference(
    #         renderer_thresh,
    #         dataloader,
    #         cfg_nerf.inference.pointcloud.max_variance,
    #         cfg_nerf.inference.pointcloud.distribution_area,
    #         cfg_nerf.trainer.n_rays,
    #         cfg_nerf.inference.pointcloud.cams,
    #         cfg_nerf.inference.pointcloud.freq,
    #         cfg_nerf.inference.pointcloud.side_margin)
    # }

    ### Finish Load OG NeRF

    ### Init SDF NeRF
    renderer_sdf = NeuSRenderer(
        renderer_nerf=renderer_thresh,
        )

    ### Fish init SDF NeRF

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
        # model=model,
        dataloader=dataloader,
        logger=logger,
        renderer=renderer_sdf,
        inferencers=inferencers_sdf,

        optimizer=optimizer,
        scheduler=scheduler, 
        
        n_rays=cfg.trainer.n_rays,
        num_epochs=cfg.trainer.num_epochs,
        iters_per_epoch=cfg.trainer.iters_per_epoch,

        dist_loss_range=cfg.trainer.dist_loss_range,
        depth_loss_range=cfg.trainer.depth_loss_range,

        eval_image_freq=cfg.log.eval_image_freq,
        eval_pointcloud_freq=cfg.log.eval_pointcloud_freq,
        save_weights_freq=cfg.log.save_weights_freq,

        metrics={},
        )

    logger.log('Beginning Training...\n')
    trainer.train()


if __name__ == '__main__':
    train()