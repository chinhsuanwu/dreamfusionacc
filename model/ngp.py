from typing import Callable, List, Union

import torch
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


class NGPradianceField(torch.nn.Module):
    """Instance-NGP radiance Field"""

    def __init__(
        self,
        aabb: Union[torch.Tensor, List[float]],
        num_dim: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geo_feat_dim: int = 31,
        n_levels: int = 16,
        log2_hashmap_size: int = 19,
        use_predict_normal: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dim = num_dim
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.use_predict_normal = use_predict_normal

        self.geo_feat_dim = geo_feat_dim
        per_level_scale = 1.4472692012786865

        self.mlp_encoder = tcnn.Encoding(
            n_input_dims=num_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            },
        )
        self.mlp_sigma = tcnn.Network(
            n_input_dims=32,
            n_output_dims=4,
            network_config={
                # "otype": "FullyFusedMLP",
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 32,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_normal = tcnn.Network(
            n_input_dims=32,
            n_output_dims=3,
            network_config={
                # "otype": "FullyFusedMLP",
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 32,
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

        h = self.mlp_sigma(x)

        density_before_activation, base_mlp_out = torch.split(h, [1, 3], dim=-1)

        # add density bias to pre-activation as stated in Magic3D
        density_before_activation = density_before_activation + density_bias

        density = (
            self.density_activation(density_before_activation) * selector[..., None]
        )

        if return_feat:
            return density, base_mlp_out, x
        else:
            return density

    def _query_rgb(self, base_mlp_out):
        rgb = torch.sigmoid(base_mlp_out)
        return rgb

    def _query_normal(self, embedding):
        return self.mlp_normal(embedding)

    def _finite_difference_normals_approximator(self, positions, bound=2, epsilon=1e-2):
        # finite difference
        # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
        pos_x = positions + torch.tensor(
            [[epsilon, 0.00, 0.00]], device=positions.device
        )
        dist_dx_pos = self.query_density(pos_x.clamp(-bound, bound), bound)[0]
        pos_y = positions + torch.tensor(
            [[0.00, epsilon, 0.00]], device=positions.device
        )
        dist_dy_pos = self.query_density(pos_y.clamp(-bound, bound), bound)[0]
        pos_z = positions + torch.tensor(
            [[0.00, 0.00, epsilon]], device=positions.device
        )
        dist_dz_pos = self.query_density(pos_z.clamp(-bound, bound), bound)[0]

        neg_x = positions + torch.tensor(
            [[-epsilon, 0.00, 0.00]], device=positions.device
        )
        dist_dx_neg = self.query_density(neg_x.clamp(-bound, bound), bound)[0]
        neg_y = positions + torch.tensor(
            [[0.00, -epsilon, 0.00]], device=positions.device
        )
        dist_dy_neg = self.query_density(neg_y.clamp(-bound, bound), bound)[0]
        neg_z = positions + torch.tensor(
            [[0.00, 0.00, -epsilon]], device=positions.device
        )
        dist_dz_neg = self.query_density(neg_z.clamp(-bound, bound), bound)[0]

        return torch.cat(
            [
                0.5 * (dist_dx_pos - dist_dx_neg) / epsilon,
                0.5 * (dist_dy_pos - dist_dy_neg) / epsilon,
                0.5 * (dist_dz_pos - dist_dz_neg) / epsilon,
            ],
            dim=-1,
        )

    def forward(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor = None,
    ):
        if self.use_viewdirs and (directions is not None):
            assert (
                positions.shape == directions.shape
            ), f"{positions.shape} v.s. {directions.shape}"

            density, base_mlp_out, embedding = self.query_density(
                positions, return_rgb=True, return_feat=True
            )
            rgb = self._query_rgb(base_mlp_out)
            if self.use_predict_normal:
                normal = self._query_normal(embedding)
            else:
                normal = self._finite_difference_normals_approximator(positions)

        return rgb, density, normal
