import numpy as np
import torch
import torchmetrics
import time
import math as m
import matplotlib.pyplot as plt
import open3d as o3d
import frnn

from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
from typing import Union

import nerf.losses as losses

from nerf.nets import NeRF
from nerf.logger import Logger
from nerf.inference import render_image, render_invdepth_thresh, generate_pointcloud
from nerf.metrics import MetricWrapper
from nerf.misc import color_depthmap


# def gaussian(offsets, sigma):

#     # offsets [N, M, 3]
#     # outputs [N, M]
    
#     distance = torch.linalg.norm(offsets, dim=-1)


def laplace(offsets, b):
    distance = torch.linalg.norm(offsets, dim=-1)
    laplace = 1 / (2 * b) * torch.exp(-distance/b)
    return laplace

class PreTrainer(object):
    def __init__(
        self,
        model,
        sparse_points,
        optimizer,
        logger,
        ):
        self.model = model
        self.sparse_points = sparse_points
        self.logger = logger

        self.batch_size = 16384
        self.sub_batch_size = 128

        self.max_dist = 0.02

        self.max_density = 1000
        self.min_density = 0.01

        self.num_epochs = 100
        self.iters_per_epoch = 100

        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()

        self.iter = 0

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_epoch(self.iters_per_epoch)

            # Output recorded scalars
            self.logger.log(f'Iteration: {self.iter}')
            for key, val in self.logger.scalars.items():
                moving_avg = np.mean(np.array(val[-self.iters_per_epoch:])).item()
                self.logger.log(f'Scalar: {key} - Value: {moving_avg:.6f}')
            self.logger.log('')

        self.logger.model(self.model.model, self.iter)


    def train_epoch(self, iters_per_epoch:int):
        for i in tqdm(range(iters_per_epoch)):
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = self.train_step()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            self.iter += 1

    def train_step(self):
        # get a bunch of random coordinates around sparse points

        ## select a random batch of points
        n = torch.randint(0, len(self.sparse_points), (self.batch_size,), device="cuda")
        # n = torch.randint(0, 4, (self.batch_size,), device="cuda")
        # print(n.shape, self.sparse_points.shape)
        # target_points = self.sparse_points[n]
        target_points = torch.index_select(self.sparse_points, 0, n)

        ## select a random sub-batch of cooridinates around that point
        offsets = (torch.rand((self.batch_size, self.sub_batch_size, 3), device="cuda") - 0.5) * 2 * self.max_dist
        # print(offsets.shape, target_points.shape)
        coordinate_samples = target_points[:, None, :] + offsets

        ## knn
        # print(target_points.shape, coordinate_samples.shape)
        # exit()
        l1 = torch.LongTensor((self.batch_size,)).to("cuda")
        l2 = torch.LongTensor((self.batch_size * self.sub_batch_size,)).to("cuda")
        # frnn.frnn_grid_points(target_points.view(1, -1, 3), coordinate_samples.view(1, -1, 3), l1, l2, K=1, r=0.02)
        distances, _, _, _ = frnn.frnn_grid_points(coordinate_samples.view(1, -1, 3), target_points.view(1, -1, 3), l2, l1, K=1, r=0.02)
        distances = distances.view(self.batch_size, self.sub_batch_size)
        distances = torch.sqrt(distances)
        # print(distances[0])
        # print(distances.view(self.batch_size, self.sub_batch_size).shape)
        # exit()
        # print(distances - torch.linalg.norm(offsets, dim=-1))
        # print(distances)
        # print(torch.linalg.norm(offsets, dim=-1))
        # exit()

        ## get the target density based on distrubution X (start with abitary density target)
        # target_density = laplace(offsets, 0.005) * self.max_density + self.min_density
        target_density = torch.exp(-torch.linalg.norm(offsets, dim=-1) / 0.0025) * self.max_density + self.min_density
        # target_density = distances
        # target_density = torch.exp(-distances / 0.0025) * self.max_density + self.min_density
        # print(torch.amax(target_density), torch.amin(target_density), torch.mean(target_density))
        # print(torch.amax(target_density), torch.amin(target_density))

        ## sample model density
        pred_density = self.model.density(coordinate_samples)
        # print(coordinate_samples.amax(), coordinate_samples.amin(), coordinate_samples.dtype)
        # print(torch.amax(pred_density), torch.amin(pred_density))
        # print()

        ## train towards that using a MSE loss
        loss = F.mse_loss(pred_density, target_density)
        # loss = F.l1_loss(pred_density, target_density)

        self.logger.scalar('loss', loss, self.iter)
        self.logger.scalar('min_pred', pred_density.amin(), self.iter)
        self.logger.scalar('max_pred', pred_density.amax(), self.iter)

        return loss