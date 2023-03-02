import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import math as m
import matplotlib.pyplot as plt

from typing import Union, Optional
from dataclasses import dataclass

from nerf.nets import NeRFNetwork, NeRFCoordinateWrapper
from nerf.render import get_rays, get_uniform_z_vals, efficient_sampling, get_sample_points, render_rays_log, regularize_weights, sample_pdf

# from sdf.model.density import LaplaceDensity
# from sdf.model.ray_sampler import ErrorBoundSampler


def neus_render_rays_log(sdf, color, z_vals, z_vals_logs, s, grad_theta, rays_d, cos_anneal_ratio):
    N_rays, N_samples = z_vals.shape[:2]

    # sdf_scaled = sdf * s

    # alpha = (torch.sigmoid(sdf_scaled[..., :-1]) - torch.sigmoid(sdf_scaled[..., 1:])) / torch.sigmoid(sdf_scaled[..., :-1])

    true_cos = (rays_d[:, None, :].expand(grad_theta.shape) * grad_theta).sum(-1)

    iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

    z_vals_mid = (z_vals[:, :-1] + z_vals[:, 1:]) / 2  # [N_rays, N_samples]
    dists = z_vals[:, 1:] - z_vals[:, :-1]  # [N_rays, N_samples]

    # Estimate signed distances at section points
    estimated_next_sdf = sdf[..., :-1] + iter_cos[..., :-1] * dists * 0.5
    estimated_prev_sdf = sdf[..., :-1] - iter_cos[..., :-1] * dists * 0.5

    prev_cdf = torch.sigmoid(estimated_prev_sdf * s)
    next_cdf = torch.sigmoid(estimated_next_sdf * s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).reshape(dists.shape).clip(0.0, 1.0)

    transmittance = torch.cumprod(1 - alpha, dim=-1)
    transmittance = torch.cat((transmittance.new_zeros((transmittance.shape[0], 1)), transmittance), dim=-1)

    alpha = torch.cat((alpha, alpha.new_zeros((alpha.shape[0], 1))), dim=-1).clip(0.0, 1.0)

    weight = transmittance * alpha

    # print("weight: ", torch.amax(torch.sum(weight, dim=-1)), torch.amin(torch.sum(weight, dim=-1)))
    # print("alpha: ", torch.amax(alpha), torch.amin(alpha))

    pixel = torch.sum(weight[..., None] * color, -2)  # [N_rays, 3]
    invdepth = torch.sum(weight / z_vals, -1)  # [N_rays]

    return pixel, invdepth, weight


# @torch.no_grad()
def neus_efficient_sampling(
    model,
    rays_o,
    rays_d,
    steps_firstpass,
    z_bounds,
    steps_importance,
    alpha_importance,
    s,
    cos_anneal_ratio):
    n_rays = rays_o.shape[0]
    z_vals_log, z_vals = get_uniform_z_vals(steps_firstpass, z_bounds, n_rays)

    xyzs, _ = get_sample_points(rays_o, rays_d, z_vals)

    grad_theta = model.gradient(xyzs)

    with torch.no_grad():
        sdf = model.sdf(xyzs)

        # print(rays_d.shape, grad_theta.shape)
        true_cos = (rays_d[:, None, :].expand(grad_theta.shape) * grad_theta).sum(-1)
        # print(true_cos.shape, sdf.shape)

        # print(cos_anneal_ratio)

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        z_vals_mid = (z_vals[:, :-1] + z_vals[:, 1:]) / 2  # [N_rays, N_samples]
        dists = z_vals[:, 1:] - z_vals[:, :-1]  # [N_rays, N_samples]

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[..., :-1] + iter_cos[..., :-1] * dists * 0.5
        estimated_prev_sdf = sdf[..., :-1] - iter_cos[..., :-1] * dists * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * s)
        next_cdf = torch.sigmoid(estimated_next_sdf * s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(dists.shape).clip(0.0, 1.0)

        # print(torch.amax(sdf), torch.amin(sdf))

        # sdf_scaled = sdf * s

        # alpha = (torch.sigmoid(sdf_scaled[..., :-1]) - torch.sigmoid(sdf_scaled[..., 1:])) / torch.sigmoid(sdf_scaled[..., :-1])

        transmittance = torch.cumprod(1 - alpha, dim=-1)
        transmittance = torch.cat((transmittance.new_zeros((transmittance.shape[0], 1)), transmittance), dim=-1)

        alpha = torch.cat((alpha, alpha.new_zeros((alpha.shape[0], 1))), dim=-1)

        weight = transmittance * alpha

        # print("weight_sampling: ", torch.sum(weight, dim=-1))


        weight_re = regularize_weights(weight, alpha_importance, steps_firstpass)

        # print(weight_re)

        # print("weight_re_sampling: ", torch.sum(weight_re, dim=-1))
        # print(z_vals_log)
        # exit()

        z_vals_log_fine = sample_pdf(z_vals_log, weight_re, steps_importance)
        z_vals_fine = torch.pow(10, z_vals_log_fine)

    return z_vals_log_fine, z_vals_fine, grad_theta


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device='cuda') * torch.exp(self.variance * 10.0)


class NeusRender(nn.Module):
    def __init__(self,
        model,
        steps_firstpass,
        z_bounds,
        steps_importance,
        alpha_importance):
        super().__init__()

        self.model = model
        self.steps_firstpass = steps_firstpass
        self.z_bounds = z_bounds
        self.steps_importance = steps_importance
        self.alpha_importance = alpha_importance

        # self.s = torch.nn.Parameter(torch.full((1,), fill_value=3.0), requires_grad=True)
        self.deviation_network = SingleVarianceNetwork(0.3)

    def render(self, n, h, w, K, E, bg_color, cos_anneal_ratio=0.0):
        rays_o, rays_d = get_rays(h, w, K, E)
        inv_s = self.deviation_network(torch.zeros([1, 3], device='cuda'))[:, :1].clip(1e-6, 1e6)
        z_vals_log, z_vals, grad_theta = neus_efficient_sampling(
            self.model, rays_o, rays_d,
            self.steps_firstpass, self.z_bounds, self.steps_importance, self.alpha_importance, inv_s, cos_anneal_ratio)

        xyzs, dirs = get_sample_points(rays_o, rays_d, z_vals)
        n_expand = n[:, None].expand(-1, z_vals.shape[-1])

        sdf, color = self.model(xyzs, dirs, n_expand)

        # print(torch.amax(sdf), torch.amin(sdf))

        # plt.plot(sdf[0].detach().cpu().numpy())
        # plt.show()

        pixel, invdepth, weight = neus_render_rays_log(sdf, color, z_vals, z_vals_log, inv_s, grad_theta, rays_d, cos_anneal_ratio)

        if bg_color is not None:
            pixel = pixel + (1 - torch.sum(weight, dim=-1)[..., None]) * bg_color

        # if self.training:
        #     grad_theta = self.model.gradient(xyzs)
        # else:
        #     grad_theta = None

        aux_outputs = {}
        aux_outputs['invdepth'] = invdepth.detach()
        aux_outputs['sdf'] = sdf.detach()
        aux_outputs['z_vals'] = z_vals.detach()

        return pixel, weight, grad_theta, aux_outputs