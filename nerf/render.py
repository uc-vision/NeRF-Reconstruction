import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m

from typing import Union, Optional
from dataclasses import dataclass

from nerf.nets import NeRFNetwork, NeRFCoordinateWrapper


## Calculate density distribiution along ray using forward only first pass to inform second pass
@torch.no_grad()
def get_uniform_z_vals(steps:list[int], z_bounds:list[int], n_rays:int):
    z_vals_log = torch.empty((0), device='cuda')
    for i in range(len(steps)):
        z_vals_start = m.log10(z_bounds[i])
        z_vals_end = m.log10(z_bounds[i+1])-(m.log10(z_bounds[i+1])-m.log10(z_bounds[i]))/steps[i]
        z_vals_log = torch.cat((z_vals_log, torch.linspace(z_vals_start, z_vals_end, steps[i], device='cuda')))
    z_vals_log = z_vals_log.expand(n_rays, -1)
    z_vals = torch.pow(10, z_vals_log)
    return z_vals_log, z_vals


@torch.no_grad()
def regularize_weights(weights, alpha, steps):
    weights = torch.cat([weights.new_zeros((weights.shape[0], 1)), weights, weights.new_zeros((weights.shape[0], 1))], dim=-1)
    weights = 1/2 * (torch.maximum(weights[..., :-2], weights[..., 1:-1]) + torch.maximum(weights[..., 1:-1], weights[..., 2:]))
    weights = weights + alpha / weights.shape[-1]
    c = 0
    for i in range(len(steps)):
        weights[:, c:c+steps[i]] *= steps[i] / sum(steps)
        c += steps[i]
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)  # [N_rays, N_samples-1]
    return weights


@torch.no_grad()
def sample_pdf(z_vals, weights, steps):

    N_rays, N_samples = z_vals.shape

    pdf = weights  # [N_rays, N_samples]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, N_samples]

    # print(cdf.shape)

    z_vals_mid = (z_vals[:, :-1] + z_vals[:, 1:]) / 2  # [N_rays, N_samples]

    # Take uniform samples
    u = torch.linspace(0, 1-1/steps, steps, device=weights.device)  # [N_rays, steps]
    u = u[None, :].expand(N_rays, -1)
    u = u + torch.rand([N_rays, steps], device=weights.device)/steps  # [N_rays, N_samples]
    u = u * (cdf[:, -2, None] - cdf[:, 1, None])
    u = u + cdf[:, 1, None]

    inds = torch.searchsorted(cdf[:, 1:-2], u, right=True) + 1  # [N_rays, steps]

    cdf_below = torch.gather(input=cdf, dim=1, index=inds-1)
    cdf_above = torch.gather(input=cdf, dim=1, index=inds)
    t = (u - cdf_below) / (cdf_above - cdf_below)

    z_vals_mid_below = torch.gather(input=z_vals_mid, dim=1, index=inds-1)
    z_vals_mid_above = torch.gather(input=z_vals_mid, dim=1, index=inds)
    z_vals_im = z_vals_mid_below + (z_vals_mid_above - z_vals_mid_below) * t

    z_vals_fine = z_vals_im

    return z_vals_fine


@torch.no_grad()
def efficient_sampling(
    models:NeRFCoordinateWrapper,
    rays_o,
    rays_d,
    steps_firstpass,
    z_bounds,
    steps_importance,
    alpha_importance,):
    n_rays = rays_o.shape[0]
    z_vals_log, z_vals = get_uniform_z_vals(steps_firstpass, z_bounds, n_rays)

    xyzs, _ = get_sample_points(rays_o, rays_d, z_vals)

    sigma = models.density(xyzs)

    delta = z_vals_log.new_zeros(sigma.shape)  # [N_rays, N_samples]
    delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [N_rays, N_samples]

    alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

    weights_re = regularize_weights(weights, alpha_importance, steps_firstpass)

    z_vals_log_fine = sample_pdf(z_vals_log, weights_re, steps_importance)
    z_vals_fine = torch.pow(10, z_vals_log_fine)

    return z_vals_log_fine, z_vals_fine


def get_rays(h, w, K, E):
    """ Ray generation for Oliver's calibration format """
    dirs = torch.stack([w+0.5, h+0.5, torch.ones_like(w)], -1)  # [N_rays, 3]
    dirs = torch.inverse(K) @ dirs[:, :, None]  # [N_rays, 3, 1]

    rays_d = E[:, :3, :3] @ dirs  # [N_rays, 3, 1]
    rays_d = torch.squeeze(rays_d, dim=2)  # [N_rays, 3]

    rays_o = E[:, :3, -1].expand(rays_d.shape[0], -1)

    return rays_o, rays_d

    
def get_sample_points(rays_o, rays_d, z_vals):
    N_rays, N_samples = z_vals.shape

    query_points = rays_d.new_zeros((N_rays, N_samples, 3))  # [N_rays, N_samples, 3]
    query_points = rays_o[:, None, :].expand(-1, N_samples, -1) + rays_d[:, None, :].expand(-1, N_samples, -1) * z_vals[..., None]  # [N_rays, N_samples, 3]

    norm = torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    query_dirs = rays_d / norm
    query_dirs = query_dirs[:, None, :].expand(-1, N_samples, -1)

    return query_points, query_dirs
    

def render_rays_log(sigma, color, z_vals, z_vals_log):
    N_rays, N_samples = z_vals.shape[:2]

    delta = z_vals_log.new_zeros([N_rays, N_samples])  # [N_rays, N_samples]
    delta[:, :-1] = (z_vals_log[:, 1:] - z_vals_log[:, :-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [N_rays, N_samples]
    alpha_shift = torch.cat([alpha.new_zeros((alpha.shape[0], 1)), alpha], dim=-1)[:, :-1]  # [N_rays, N_samples]
    weight = alpha * torch.cumprod(1 - alpha_shift, dim=-1)  # [N_rays, N_samples]

    pixel = torch.sum(weight[..., None] * color, -2)  # [N_rays, 3]
    invdepth = torch.sum(weight / z_vals, -1)  # [N_rays]

    return pixel, invdepth, weight



def render_nerf(
    models:NeRFCoordinateWrapper,
    n:torch.LongTensor,
    h:torch.LongTensor,
    w:torch.LongTensor,
    K:torch.Tensor,
    E:torch.Tensor,
    steps_firstpass:Union[int, list[int]]=None,
    z_bounds:list[int]=None,
    steps_importance:int=None,
    alpha_importance:float=None,
    bg_color:Optional[torch.Tensor]=None):
    
    ray_o, ray_d = get_rays(h, w, K, E)
    z_val_log, z_val = efficient_sampling(
        models, ray_o, ray_d,
        steps_firstpass, z_bounds, steps_importance, alpha_importance)
    xyz, dir = get_sample_points(ray_o, ray_d, z_val)
    n_expand = n[:, None].expand(-1, z_val.shape[-1])

    sigma, color = models(xyz, dir, n_expand)

    pixel, invdepth, weight = render_rays_log(sigma, color, z_val, z_val_log)
    if bg_color is not None:
        pixel = pixel + (1 - torch.sum(weight, dim=-1)[..., None]) * bg_color

    z_vals_log_norm = (z_val_log - m.log10(z_bounds[0])) / (m.log10(z_bounds[-1]) - m.log10(z_bounds[0]))

    # auxilary outputs that may be useful for inference of analysis, but not used in training
    aux_outputs = {}
    aux_outputs['invdepth'] = invdepth.detach()
    aux_outputs['z_val'] = z_val.detach()
    aux_outputs['z_val_log'] = z_val_log.detach()
    aux_outputs['sigma'] = sigma.detach()
    aux_outputs['color'] = color.detach()
    aux_outputs['xyz'] = xyz.detach()

    return pixel, weight, z_vals_log_norm, aux_outputs


class Render(object):
    def __init__(self,
        models,
        steps_firstpass:Union[int, list[int]],
        z_bounds:list[int],
        steps_importance:int,
        alpha_importance:float):
        
        self.models = models
        self.steps_firstpass = steps_firstpass
        self.z_bounds = z_bounds
        self.steps_importance = steps_importance
        self.alpha_importance = alpha_importance

    def render(self, n, h, w, K, E, bg_color=None):
        return render_nerf(
            self.models, 
            n, h, w, K, E,
            steps_firstpass=self.steps_firstpass,
            z_bounds=self.z_bounds,
            steps_importance=self.steps_importance,
            alpha_importance=self.alpha_importance,
            bg_color=bg_color)