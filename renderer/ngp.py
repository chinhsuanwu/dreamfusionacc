from typing import Callable, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import tinycudann as tcnn
except ImportError as e:
    print(
        f"Error: {e}! "
        "Please install tinycudann by: "
        "pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
    )
    exit()


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x


def safe_normalize(x: torch.Tensor):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=1e-20))


class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        use_viewdirs: bool = True,
        density_activation: Callable = lambda x: F.softplus(x - 1),
        unbounded: bool = False,
        base_resolution: int = 16,
        max_resolution: int = 4096,
        geo_feat_dim: int = 31,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        use_normal_net: bool = True,
        density_bias_scale: int = 10,
        offset_scale: float = 0.5,
        use_bkgd_net: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.use_normal_net = use_normal_net
        self.density_bias_scale = density_bias_scale
        self.offset_scale = offset_scale
        self.use_bkgd_net = use_bkgd_net

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = np.exp(
            (np.log(max_resolution) - np.log(base_resolution)) / (n_levels - 1)
        ).tolist()

        if self.use_viewdirs:
            self.direction_encoding = tcnn.Encoding(
                n_input_dims=num_dim,
                encoding_config={
                    "otype": "Composite",
                    "nested": [
                        {
                            "n_dims_to_encode": 3,
                            "otype": "SphericalHarmonics",
                            "degree": 4,
                        },
                    ],
                    # {"otype": "Identity", "n_bins": 4, "degree": 4},
                },
            )

        self.mlp_encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            },
            dtype=torch.float32,  # float16 will cause NaN issue
        )
        self.mlp_sigma = tcnn.Network(
            n_input_dims=1 + self.geo_feat_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_rgb = tcnn.Network(
            n_input_dims=(
                (self.direction_encoding.n_output_dims if self.use_viewdirs else 0)
                + 1
                + self.geo_feat_dim
            ),
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            },
        )

        if self.use_normal_net:
            self.mlp_normal = tcnn.Network(
                n_input_dims=(
                    (self.direction_encoding.n_output_dims if self.use_viewdirs else 0)
                    + 1
                    + self.geo_feat_dim
                ),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 32,
                    "n_hidden_layers": 1,
                },
            )

        if self.use_bkgd_net:
            assert self.use_viewdirs is True
            self.mlp_bkgd = tcnn.Network(
                n_input_dims=self.direction_encoding.n_output_dims,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 16,
                    "n_hidden_layers": 1,
                },
            )

    def density_bias(self, x, density_bias_scale: int = 10, offset_scale: int = 0.5):
        tau = density_bias_scale * (1 - torch.sqrt((x**2).sum(-1)) / offset_scale)
        return tau[:, None]

    def query_density(self, x, return_feat: bool = False):
        density_bias = self.density_bias(x)

        if self.unbounded:
            x = contract_to_unisphere(x, self.aabb)
        else:
            aabb_min, aabb_max = torch.split(self.aabb, self.num_dim, dim=-1)
            x = (x - aabb_min) / (aabb_max - aabb_min)
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)

        x = (
            self.mlp_encoder(x.view(-1, self.num_dim))
            .view(list(x.shape[:-1]) + [1 + self.geo_feat_dim])
            .to(x)
        )

        # add density bias to pre-activation
        density_before_activation = self.mlp_sigma(x) + density_bias

        density = (
            self.density_activation(density_before_activation) * selector[..., None]
        )

        if return_feat:
            return density, x, density_before_activation
        else:
            return density

    def _compute_embedding(self, dir, embedding):
        # tcnn requires directions in the range [0, 1]
        if self.use_viewdirs:
            dir = (dir + 1.0) / 2.0
            d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
            return torch.cat([d, embedding.view(-1, 1 + self.geo_feat_dim)], dim=-1)
        else:
            return embedding.view(-1, 1 + self.geo_feat_dim)

    def _query_rgb(self, embedding):
        return self.mlp_rgb(embedding)

    def _query_normal(self, embedding):
        return self.mlp_normal(embedding)

    def query_bkgd(self, dir):
        dir = (dir + 1.0) / 2.0
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        return self.mlp_bkgd(d)

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
        shading: str = "albedo",
        light_direction: torch.Tensor = None,
    ):
        if shading == "albedo":
            density, embedding, _ = self.query_density(positions, return_feat=True)
            embedding = self._compute_embedding(directions, embedding)
            rgb = self._query_rgb(embedding)
            normal = None
        else:
            with torch.enable_grad():
                positions.requires_grad_(True)

                density, embedding, density_before_activation = self.query_density(
                    positions, return_feat=True
                )
                embedding = self._compute_embedding(directions, embedding)
                rgb = self._query_rgb(embedding)

                if self.use_normal_net:
                    normal = self._query_normal(embedding)
                else:
                    # https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/fields/base_field.py#L87
                    normal = -torch.autograd.grad(
                        density_before_activation,
                        positions,
                        grad_outputs=torch.ones_like(density_before_activation),
                        retain_graph=True,
                    )[0]
                    normal = safe_normalize(normal).nan_to_num()

            ambient_ratio = 0.1 + 0.9 * np.random.rand()
            light_direction = light_direction.to(normal.dtype)
            lambertian = ambient_ratio + (1 - ambient_ratio) * (
                normal @ light_direction / (light_direction**2).sum() ** (1 / 2)
            ).clamp(min=0).unsqueeze(-1)

            if shading == "textureless":
                rgb = lambertian.tile(1, 3)
            elif shading == "lambertian":
                rgb = rgb * lambertian
            elif shading == "normal":
                rgb = (normal + 1) / 2
            else:
                NotImplementedError()

        return rgb, density, normal

    def get_params(self, lr):
        params = [
            {"params": self.mlp_encoder.parameters(), "lr": lr},
            {"params": self.mlp_sigma.parameters(), "lr": lr},
            {"params": self.mlp_rgb.parameters(), "lr": lr},
        ]

        if self.use_viewdirs:
            params.append(
                {"params": self.direction_encoding.parameters(), "lr": lr},
            )
        if self.use_normal_net:
            params.append(
                {"params": self.mlp_normal.parameters(), "lr": lr},
            )
        if self.use_bkgd_net:
            # background net has lower learning rate
            params.append(
                {"params": self.mlp_bkgd.parameters(), "lr": lr / 10},
            )

        return params
