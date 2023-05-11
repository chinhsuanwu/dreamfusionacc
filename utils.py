import random
from typing import Optional

import numpy as np
import torch
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.volrend import *

from dataset.utils import Rays, namedtuple_map, rand_poses


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def custom_rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    n_rays: int,
    # radiance field
    rgb_sigma_normal_fn: Optional[Callable] = None,
    # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
    shading: str = "albedo",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use custom rendering functoin so we can render normals for orientation loss.
    Please refer to nerfacc.volrend.rendering()
    """

    if rgb_sigma_normal_fn is None:
        raise ValueError("`rgb_sigma_normal_fn` should be specified.")

    # Query sigma/alpha and color with gradients
    rgbs, sigmas, normals = rgb_sigma_normal_fn(t_starts, t_ends, ray_indices, shading)

    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights.
    weights, trans, alphas = render_weight_from_density(
        t_starts,
        t_ends,
        sigmas,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, values=rgbs, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    depths = accumulate_along_rays(
        weights,
        values=(t_starts + t_ends)[..., None] / 2.0,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )
    depths = depths / opacities.clamp_min(torch.finfo(rgbs.dtype).eps)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, normals, weights


def render_image_with_occgrid(
    # scene
    radiance_field: torch.nn.Module,
    estimator: OccGridEstimator,
    rays: Rays,
    # rendering options
    near_plane: float = 0.0,
    far_plane: float = 1e10,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    chunk_size: int = 8192,
    shading: str = "albedo",
    use_bkgd_net: bool = False,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays)
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        sigmas = radiance_field.query_density(positions)
        return sigmas.squeeze(-1)

    def rgb_sigma_normal_fn(t_starts, t_ends, ray_indices, shading):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

        # point light shading, [3,]
        light_direction = rand_poses(
            1, radius_range=[0.8, 1.5], theta_range=[0, 60], device=positions.device
        )[0][-1, :3, -1]

        rgbs, sigmas, normals = radiance_field(
            positions, t_dirs, shading, light_direction
        )
        return rgbs, sigmas.squeeze(-1), normals

    results = []
    chunk = torch.iinfo(torch.int32).max if radiance_field.training else chunk_size

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        ray_indices, t_starts, t_ends = estimator.sampling(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        if use_bkgd_net:
            render_bkgd = radiance_field.query_bkgd(chunk_rays.viewdirs)

        # use customized rendering function to render normal
        rgb, opacity, depth, normal, weight = custom_rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_normal_fn=rgb_sigma_normal_fn,
            render_bkgd=render_bkgd,
            shading=shading,
        )

        if radiance_field.training and normal is not None and shading != "albedo":
            loss_orient = (
                weight.detach()
                * (normal * chunk_rays.viewdirs[ray_indices]).sum(-1).clamp(min=0) ** 2
            )
            loss_orient = loss_orient.mean().unsqueeze(-1)
        else:
            loss_orient = torch.zeros((1, 1))  # dummy

        chunk_results = [rgb, opacity, depth, loss_orient, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, loss_orient, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]

    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        loss_orient.mean().unsqueeze(-1) if normal is not None else None,
        sum(n_rendering_samples),
    )
