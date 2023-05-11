"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections

import numpy as np
import torch
import torch.nn.functional as F

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(
    size,
    device,
    radius_range=[1.2, 2],
    theta_range=[0, 120],
    phi_range=[0, 360],
    return_dirs=False,
    angle_overhead=30,
    angle_front=60,
    jitter=False,
    uniform_sphere_rate=0.5,
):
    """generate random poses from an orbit camera
    Original code: https://github.com/ashawkey/stable-dreamfusion

    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    """

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = (
        torch.rand(size, device=device) * (radius_range[1] - radius_range[0])
        + radius_range[0]
    )

    if np.random.rand() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack(
                [
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                    torch.rand(size, device=device),
                    (torch.rand(size, device=device) - 0.5) * 2.0,
                ],
                dim=-1,
            ),
            p=2,
            dim=1,
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = (
            torch.rand(size, device=device) * (theta_range[1] - theta_range[0])
            + theta_range[0]
        )
        phis = (
            torch.rand(size, device=device) * (phi_range[1] - phi_range[0])
            + phi_range[0]
        )

        centers = torch.stack(
            [
                radius * torch.sin(thetas) * torch.sin(phis),
                radius * torch.cos(thetas),
                radius * torch.sin(thetas) * torch.cos(phis),
            ],
            dim=-1,
        )  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(centers - targets)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(
        torch.cross(right_vector, forward_vector, dim=-1) + up_noise
    )

    poses = (
        torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    )
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


def circle_poses(
    device,
    radius=1.25,
    theta=60,
    phi=0,
    return_dirs=False,
    angle_overhead=30,
    angle_front=60,
):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = safe_normalize(centers)
    up_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs
