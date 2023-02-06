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
from nerf.inference import ImageInference, InvdepthThreshInference, PointcloudInference

# from misc import configurator


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def train(cfg : DictConfig) -> None:


    logger = Logger(
        root_dir=cfg.log.root_dir,
        cfg=cfg,
        )

    logger.log('Initiating Dataloader...')
    dataloader = CameraGeometryLoader(
        scan_paths=cfg.scan.scan_paths,
        scan_pose_paths=cfg.scan.scan_pose_paths,
        frame_ranges=cfg.scan.frame_ranges,
        frame_strides=cfg.scan.frame_strides,
        image_scale=cfg.scan.image_scale,
        )

    logger.log('Initilising Model...')
    model = NeRFNetwork(
        N = dataloader.images.shape[0],
        encoding_precision=cfg.nets.encoding.precision,
        encoding_n_levels=cfg.nets.encoding.n_levels,
        encoding_n_features_per_level=cfg.nets.encoding.n_features_per_level,
        encoding_log2_hashmap_size=cfg.nets.encoding.log2_hashmap_size,
        geo_feat_dim=cfg.nets.sigma.geo_feat_dim,
        sigma_hidden_dim=cfg.nets.sigma.hidden_dim,
        sigma_num_layers=cfg.nets.sigma.num_layers,
        encoding_dir_precision=cfg.nets.encoding_dir.precision,
        encoding_dir_encoding=cfg.nets.encoding_dir.encoding,
        encoding_dir_degree=cfg.nets.encoding_dir.degree,
        latent_embedding_dim=cfg.nets.latent_embedding.features,
        color_hidden_dim=cfg.nets.color.hidden_dim,
        color_num_layers=cfg.nets.color.num_layers,
    ).to('cuda')

    transform = Transform(translation=-dataloader.translation_center).to('cuda')

    model_coord = NeRFCoordinateWrapper(
        model=model,
        transform=transform,
        # transform=None,
        inner_bound=cfg.scan.inner_bound,
        outer_bound=cfg.scan.outer_bound,
    ).to('cuda')

    logger.log('Initiating Optimiser...')
    optimizer = torch.optim.Adam([
            {'name': 'encoding', 'params': list(model.encoder.parameters()), 'lr': cfg.optimizer.encoding.lr},
            {'name': 'latent_emb', 'params': [model.latent_emb], 'lr': cfg.optimizer.latent_emb.lr},
            {'name': 'net', 'params': list(model.sigma_net.parameters()) + list(model.color_net.parameters()), 'weight_decay': cfg.optimizer.net.weight_decay, 'lr': cfg.optimizer.net.lr},
        ], betas=cfg.optimizer.betas, eps=cfg.optimizer.eps)

    if cfg.scheduler == 'step':
        lmbda = lambda x: 1
    elif cfg.scheduler == 'exp_decay':
        lmbda = lambda x: 0.1**(x/(cfg.trainer.num_epochs*cfg.trainer.iters_per_epoch))
    else:
        raise ValueError
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)

    metrics = {
        "eval_lpips": LPIPSWrapper(),
        "eval_ssim": SSIMWrapper(),
        "eval_psnr": PSNRWrapper(),
    }

    renderer = Render(
        models=model_coord,
        steps_firstpass=cfg.renderer.steps,
        z_bounds=cfg.renderer.z_bounds,
        steps_importance=cfg.renderer.importance_steps,
        alpha_importance=cfg.renderer.alpha,
    )

    renderer_thresh = Render(
        models=model_coord,
        steps_firstpass=cfg.renderer_thresh.steps,
        z_bounds=cfg.renderer_thresh.z_bounds,
        steps_importance=cfg.renderer_thresh.importance_steps,
        alpha_importance=cfg.renderer_thresh.alpha,
    )

    if cfg.inference.image.image_num == 'middle':
        rigs_num = dataloader.index_mapping.get_num_rigs()
        inference_image_num = int(dataloader.index_mapping.src_to_idx(0, rigs_num // 2, 3).item())
    else:
        inference_image_num = cfg.inference.image.image_num

    inferencers = {
        "image": ImageInference(
            renderer, dataloader, cfg.trainer.n_rays, inference_image_num),
        "invdepth_thresh": InvdepthThreshInference(
            renderer_thresh, dataloader, cfg.trainer.n_rays, inference_image_num),
        "pointcloud": PointcloudInference(
            renderer_thresh,
            dataloader,
            cfg.inference.pointcloud.max_variance,
            cfg.inference.pointcloud.distribution_area,
            cfg.trainer.n_rays,
            cfg.inference.pointcloud.cams,
            cfg.inference.pointcloud.freq,
            cfg.inference.pointcloud.side_margin)
    }
    
    logger.log('Initiating Trainer...')
    trainer = NeRFTrainer(
        model=model,
        dataloader=dataloader,
        logger=logger,
        renderer=renderer,
        inferencers=inferencers,

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

        metrics=metrics,
        )

    logger.log('Beginning Training...\n')
    trainer.train()


if __name__ == '__main__':
    train()