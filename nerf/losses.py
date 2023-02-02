import numpy as np
import torch
import math as m

from torch import nn
from torch.nn import functional as F

## Losses and Metrics
def psnr(rgb_, rgb):
    mse = F.mse_loss(rgb_, rgb)
    return -10.0 * m.log10(mse)

def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse < 1e-5:
        mse = 1e-5
    return -10.0 * m.log10(mse)

def criterion_dist(weights, z_vals):
    # z_vals_s = (z_vals + torch.min(z_vals)) / (torch.max(z_vals) - torch.min(z_vals))
    w = torch.bmm(weights[:, :, None], weights[:, None, :])
    s = torch.abs(z_vals[:, :, None] - z_vals[:, None, :])
    loss = w * s
    loss = torch.mean(torch.sum(loss, dim=[1, 2]))
    return loss

def criterion_rgb(rgb_, rgb):
    loss = F.huber_loss(rgb_, rgb, delta=0.1)
    # loss = F.mse_loss(rgb_, rgb)
    return loss