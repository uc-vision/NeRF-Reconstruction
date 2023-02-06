
import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn

from typing import Optional, Union


class NeRF(nn.Module):
    def forward(self, x, d, n):
        pass

    def density(self, x):
        pass


class NeRFNetwork(NeRF):
    def __init__(
        self,
        N,
        encoding_precision,
        encoding_n_levels,
        encoding_n_features_per_level,
        encoding_log2_hashmap_size,
        geo_feat_dim,
        sigma_hidden_dim,
        sigma_num_layers,
        encoding_dir_precision,
        encoding_dir_encoding,
        encoding_dir_degree,
        latent_embedding_dim,
        color_hidden_dim,
        color_num_layers):
        super().__init__()

        if encoding_precision == 'float16':
            self.encoding_precision = torch.float16
        elif encoding_precision == 'float32':
            self.encoding_precision = torch.float32
        else:
            return ValueError

        # hastable encoding
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": encoding_n_levels,
                "n_features_per_level": encoding_n_features_per_level,
                "log2_hashmap_size": encoding_log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
            dtype=self.encoding_precision
        )

        # sigma network
        self.sigma_net = tcnn.Network(
            n_input_dims=encoding_n_levels*encoding_n_features_per_level,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": sigma_hidden_dim,
                "n_hidden_layers": sigma_num_layers - 1,
            },
        )

        # directional encoding
        if encoding_dir_degree != 0:
            if encoding_dir_precision == 'float16':
                self.encoding_dir_precision = torch.float16
            elif encoding_dir_precision == 'float32':
                self.encoding_dir_precision = torch.float32
            else:
                return ValueError

            self.encoder_dir = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": encoding_dir_encoding,
                    "degree": encoding_dir_degree,
                },
                dtype=self.encoding_dir_precision,
            )
            self.directional_dim = self.encoder_dir.n_output_dims
        else:
            self.directional_dim = 0

        # camera latent embedding
        self.N = N
        if latent_embedding_dim != 0:
            self.latent_emb_dim = latent_embedding_dim
            self.latent_emb = nn.Parameter(torch.zeros(self.N, self.latent_emb_dim, device='cuda'), requires_grad=True)
        else:
            self.latend_emb_dim = 0

        # color network
        self.in_dim_color = geo_feat_dim + self.directional_dim + self.latent_emb_dim
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": color_hidden_dim,
                "n_hidden_layers": color_num_layers - 1,
            },
        )

    # @torch.no_grad()
    def forward(self, x, d, n):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)
        d = d.reshape(-1, 3)
        n = n.reshape(-1)

        # sigma
        x_hashtable = self.encoder(x)

        sigma_network_output = self.sigma_net(x_hashtable)

        sigma = F.relu(sigma_network_output[..., 0])
        geo_feat = sigma_network_output[..., 1:]

        # color
        color_network_input = geo_feat
        if self.directional_dim != 0:
            d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_dir(d)
            color_network_input = torch.cat([color_network_input, d], dim=-1)
        if self.latent_emb_dim != 0:
            l = self.latent_emb[n]
            color_network_input = torch.cat([color_network_input, l], dim=1)

        color_network_output = self.color_net(color_network_input)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(color_network_output)
    
        sigma = sigma.reshape(*prefix)
        color = color.reshape(*prefix, -1)


        return sigma, color

    def density(self, x):

        prefix = x.shape[:-1]
        x = x.reshape(-1, 3)

        x_hashtable = self.encoder(x)
        sigma_network_output = self.sigma_net(x_hashtable)

        sigma = F.relu(sigma_network_output[..., 0])

        sigma = sigma.reshape(*prefix)

        return sigma


class Transform(nn.Module):
    """ Homogenous transform on input coordinates """
    @torch.cuda.amp.autocast(enabled=False)
    def __init__(self, rotation=None, translation=None):
        super().__init__()

        # self.transform_matrix = transform_matrix
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.transform_matrix[:3, 3] = translation

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x_a):

        b = x_a.reshape(-1, 3)
        b1 = torch.cat([b, b.new_ones((b.shape[0], 1))], dim=1)
        b2 = (self.transform_matrix @ b1.T).T
        x_b = b2[:, :3].reshape(*x_a.shape)

        return x_b


def mipnerf360_scale(x, bound_inner, bound_outer):
    """ Performes the MIPNeRF360 coordinate warping """
    if len(x.shape) == 2:
        d = torch.linalg.norm(x, dim=-1)[..., None].expand(-1, 3)
    else:
        d = torch.linalg.norm(x, dim=-1)[..., None].expand(-1, -1, 3)
    x_warped = torch.clone(x)
    x_warped[d > bound_inner] = x_warped[d > bound_inner] * ((bound_outer - (bound_outer - bound_inner) / d[d > bound_inner]) / d[d > bound_inner])
    return x_warped


class NeRFCoordinateWrapper(NeRF):
    """ 
    Wrapper class for NeRF that performs coordinate transforms and warping
    Same input and output tensor shapes as wrapped NeRF network
    """
    def __init__(self, 
        model:NeRFNetwork,
        transform:Optional[Transform],
        inner_bound:int,
        outer_bound:int,):
        super().__init__()

        self.model = model
        self.transform = transform

        self.inner_bound = inner_bound
        self.outer_bound = outer_bound

    def forward(self, x, d, n):
        if self.transform is None:
            x_transform = x
        else:
            x_transform = self.transform(x)

        x_warped = mipnerf360_scale(x_transform, self.inner_bound, self.outer_bound)
        x_norm = (x_warped + self.outer_bound) / (2 * self.outer_bound) # to [0, 1]

        sigmas, rgbs = self.model(x_norm, d, n)

        return sigmas, rgbs

    def density(self, x):
        if self.transform is None:
            x_transform = x
        else:
            x_transform = self.transform(x)

        # x_transform_ = torch.clone(x) + self.transform.translation.to('cuda')

        # print(self.transform.transform_matrix)
        # print(x_transform.shape)
        # exit()
        # print(torch.amax(x_transform, dim=[0, 1]))
        # print(torch.amin(x_transform, dim=[0, 1]))
        # exit()

        x_warped = mipnerf360_scale(x_transform, self.inner_bound, self.outer_bound)

        # print(x_transform - x_transform_)
        # exit()
        x_norm = (x_warped + self.outer_bound) / (2 * self.outer_bound) # to [0, 1]

        sigmas = self.model.density(x_norm)

        return sigmas
