import random
from typing import Optional

import numpy as np
import torch
from dataset.utils import Rays, namedtuple_map, rand_poses

from nerfacc import OccupancyGrid, ray_marching
from nerfacc.vol_rendering import *


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Please refer to nerfacc.vol_rendering.rendering()"""

    if rgb_sigma_normal_fn is None:
        raise ValueError("`rgb_sigma_normal_fn` should be specified.")

    # Query sigma/alpha and color with gradients
    (rgbs, sigmas, normals), dirs = rgb_sigma_normal_fn(t_starts, t_ends, ray_indices)
    assert rgbs.shape[-1] == 3, "rgbs must have 3 channels, got {}".format(rgbs.shape)
    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)

    # Rendering: compute weights.
    weights = render_weight_from_density(
        t_starts,
        t_ends,
        sigmas,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )

    normals = torch.nan_to_num(normals)

    if shading != "albedo":
        ambient_ratio = 0.1

        # point light shading
        light_direction = rand_poses(
            1, radius_range=[0.8, 1.5], theta_range=[0, 60], device=normals.device
        )[0][-1, :3, -1].to(
            normals.dtype
        )  # [3,]

        lambertian = ambient_ratio + (1 - ambient_ratio) * (
            normals @ light_direction / (light_direction**2).sum() ** (1 / 2)
        ).clamp(min=0).unsqueeze(-1)

        if shading == "textureless":
            rgbs = lambertian.repeat(1, 3)
        elif shading == "lambertian":
            rgbs = rgbs * lambertian
        else:
            NotImplementedError()

    loss_orient = (
        weights.detach() * (normals * dirs).sum(-1).clamp(min=0).unsqueeze(-1) ** 2
    )
    loss_orient = loss_orient.mean().unsqueeze(-1)

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(weights, ray_indices, values=rgbs, n_rays=n_rays)
    opacities = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
    depths = accumulate_along_rays(
        weights,
        ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )
    normals = (normals + 1) / 2
    normals = accumulate_along_rays(weights, ray_indices, values=normals, n_rays=n_rays)

    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    return colors, opacities, depths, normals, loss_orient


def render_image(
    # scene
    radiance_field: torch.nn.Module,
    occupancy_grid: OccupancyGrid,
    rays: Rays,
    scene_aabb: torch.Tensor,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    shading: str = "albedo",
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
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field.query_density(positions)

    def rgb_sigma_normal_fn(t_starts, t_ends, ray_indices):
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field(positions, t_dirs), t_dirs

    results = []
    chunk = torch.iinfo(torch.int32).max

    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)

        ray_indices, t_starts, t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
        )

        # use customized rendering function to add orientation loss
        rgb, opacity, depth, normal, loss_orient = custom_rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=chunk_rays.origins.shape[0],
            rgb_sigma_normal_fn=rgb_sigma_normal_fn,
            render_bkgd=render_bkgd,
            shading=shading,
        )
        chunk_results = [rgb, opacity, depth, normal, loss_orient, len(t_starts)]
        results.append(chunk_results)
    colors, opacities, depths, normals, loss_orient, n_rendering_samples = [
        torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
        for r in zip(*results)
    ]
    return (
        colors.view((*rays_shape[:-1], -1)),
        opacities.view((*rays_shape[:-1], -1)),
        depths.view((*rays_shape[:-1], -1)),
        normals.view((*rays_shape[:-1], -1)),
        loss_orient.mean() if shading != "albedo" else 0,
        sum(n_rendering_samples),
    )
