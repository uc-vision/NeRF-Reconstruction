import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from typing import Optional, Union

# from loaders.camera_geometry_loader_re2 import IndexMapping
# from rotation import Exp_vector, Exp

from nerf.nets import NeRFNetwork, NeRFCoordinateWrapper


class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        n_levels = 20
        n_features_per_level = 2
        input_dim = n_levels * n_features_per_level
        hidden_dim = 64
        output_dim = 16

        # hastable encoding
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": 24,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=torch.float32
        )

        # sdf network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(beta=100),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, xyz):

        xyz = (xyz + 4) / 8

        # inputs = (inputs + 2) / 4  # lego
        # inputs = (inputs + 4) / 8  # vines

        x = self.encoder(xyz)
        x = self.network(x)

        return x

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            )[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = 64
        num_layers = 3
        encoding_dir_encoding = "SphericalHarmonics"
        encoding_dir_degree = 6

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": encoding_dir_encoding,
                "degree": encoding_dir_degree,
            },
            dtype=torch.float32,
        )
        self.directional_dim = self.encoder_dir.n_output_dims

        # 
        self.in_dim_color = 57

        self.network = nn.Sequential(
            nn.Linear(self.in_dim_color, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, points, normals, view_dirs, feature_vectors):
        view_dirs = (view_dirs + 1) / 2
        view_dirs = self.encoder_dir(view_dirs)

        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)

        x = self.network(rendering_input)

        return torch.sigmoid(x)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1], device="cuda") * torch.exp(self.variance * 10.0)