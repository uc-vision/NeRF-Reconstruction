import numpy as np
import torch
import torchmetrics
import time
import math as m
import matplotlib.pyplot as plt
import open3d as o3d

from tqdm import tqdm
from typing import Union

import nerf.losses as losses

from nerf.nets import NeRF
from nerf.logger import Logger
from nerf.inference import render_image, render_invdepth_thresh, generate_pointcloud
from nerf.metrics import MetricWrapper
from nerf.misc import color_depthmap


def logarithmic_scale(i, i_max, x_min, x_max):
    i_norm = i/i_max
    log_range = m.log10(x_max) - m.log10(x_min)
    x = 10**(i_norm * log_range + m.log10(x_min))
    return x

def convert_pointcloud(pointcloud_npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_npy['points'])
    pcd.colors = o3d.utility.Vector3dVector(pointcloud_npy['colors'])
    return pcd



class NeRFTrainer(object):
    def __init__(
        self,
        logger:Logger,
        dataloader,
        model:NeRF,
        renderer,
        inferencers,

        optimizer:torch.optim.Optimizer,
        scheduler,
        
        n_rays:int,
        num_epochs:int,
        iters_per_epoch:int,

        dist_loss_range:tuple[int, int],
        depth_loss_range:tuple[int, int],
        
        eval_image_freq:Union[int, str],
        eval_pointcloud_freq:Union[int, str],
        save_weights_freq:Union[int, str],

        metrics:dict[str, MetricWrapper],
        ):

        self.model = model
        self.dataloader = dataloader
        self.logger = logger
        self.renderer = renderer
        self.inferencers = inferencers

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        self.n_rays = n_rays
        self.num_epochs = num_epochs
        self.iters_per_epoch = iters_per_epoch
        self.num_iters = self.num_epochs * self.iters_per_epoch

        self.eval_image_freq = eval_image_freq if eval_image_freq != 'end' else num_epochs
        self.eval_pointcloud_freq = eval_pointcloud_freq if eval_pointcloud_freq != 'end' else num_epochs
        self.save_weights_freq =  save_weights_freq if save_weights_freq != 'end' else num_epochs

        self.dist_loss_range = dist_loss_range
        self.depth_loss_range = depth_loss_range

        self.metrics = metrics

        self.iter = 0

    def train(self):

        self.t0_train = time.time()

        for epoch in range(self.num_epochs):

            self.train_epoch(self.iters_per_epoch)

            # Output recorded scalars
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                moving_avg = np.mean(np.array(val[-self.iters_per_epoch:])).item()
                self.logger.log(f'Scalar: {key} - Value: {moving_avg:.6f}')
            self.logger.log('')
            
            # Log useful data
            if self.eval_image_freq is not None:
                if (epoch+1) % self.eval_image_freq == 0:
                    self.logger.log('Rending Image...')
                    image, invdepth = self.inferencers['image']()
                    self.logger.image('image', image.numpy(), self.iter)
                    self.logger.image('invdepth', color_depthmap(invdepth.numpy()), self.iter)

            if self.eval_image_freq is not None:
                if (epoch+1) % self.eval_image_freq == 0:
                    invdepth_thresh = self.inferencers['invdepth_thresh']()
                    self.logger.image('invdepth_thresh',
                        color_depthmap(invdepth_thresh.numpy()), self.iter)

            if self.eval_pointcloud_freq is not None:
                if (epoch+1) % self.eval_pointcloud_freq == 0:
                    self.logger.log('Generating Pointcloud...')
                    pointcloud = self.inferencers['pointcloud']()
                    self.logger.pointcloud(convert_pointcloud(pointcloud), self.iter)

            if self.save_weights_freq is not None:
                if (epoch+1) % self.save_weights_freq == 0:
                    self.logger.log('Saving Model...')
                    self.logger.model(self.model, self.iter)


    def train_epoch(self, iters_per_epoch:int):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = self.train_step()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

            self.iter += 1

    def train_step(self):
        self.model.train()
        
        n, h, w, K, E, rgb_gt, bg_color, _ = self.dataloader.get_random_batch(self.n_rays, device='cuda')

        rgb, weights, z_vals_log_s, _ = self.renderer.render(n, h, w, K, E, bg_color)

        # Calculate losses
        loss_rgb = losses.criterion_rgb(rgb, rgb_gt)
        if self.dist_loss_range[0] == 0 or self.dist_loss_range[1]== 0:
            loss_dist = 0  # non-trivial computational cost for full computation
            dist_scalar = 0
            loss = loss_rgb
        else:
            loss_dist = losses.criterion_dist(weights, z_vals_log_s)
            dist_scalar = logarithmic_scale(
                self.iter, self.num_iters, self.dist_loss_range[0], self.dist_loss_range[1])
            loss = loss_rgb + dist_scalar * loss_dist

        if self.depth_loss_range[0] == 0 or self.depth_loss_range[1] == 0:
            depth_scalar = 0
            loss_depth = 0
        else:
            depth_scalar = logarithmic_scale(
                self.iter, self.num_iters, self.depth_loss_range[0], self.depth_loss_range[1])
            loss_depth = torch.mean(weights * (1 - z_vals_log_s))
            loss = loss + depth_scalar * loss_depth

        # Log scalars
        self.logger.scalar('loss', loss, self.iter)
        self.logger.scalar('loss_rgb', loss_rgb, self.iter)
        self.logger.scalar('psnr_rgb', losses.psnr(rgb, rgb_gt), self.iter)
        self.logger.scalar('dist_scalar', dist_scalar, self.iter)
        self.logger.scalar('loss_dist', loss_dist, self.iter)
        self.logger.scalar('depth_scalar', depth_scalar, self.iter)
        self.logger.scalar('loss_depth', loss_depth, self.iter)
        
        t1 = int(time.time() - self.t0_train)
        self.logger.scalar('loss (seconds)', loss, t1)
        self.logger.scalar('loss_rgb (seconds)', loss_rgb, t1)
        self.logger.scalar('psnr_rgb (seconds)', losses.psnr(rgb, rgb_gt), t1)

        return loss