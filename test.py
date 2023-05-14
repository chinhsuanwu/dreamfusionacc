import argparse
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from nerfacc.estimators.occ_grid import OccGridEstimator
from tqdm import trange

from dataset.dreamfusion import DreamFusionLoader
from renderer.ngp import NGPradianceField
from utils import render_image_with_occgrid, set_random_seed


def read_config(fn: str = None):
    import yaml
    from attrdict import AttrDict

    with open(fn, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config = AttrDict(config)
    str2list = lambda s: [float(item) for item in s.split(",")]
    config.aabb = str2list(config.aabb)
    config.shading_sample_prob = str2list(config.shading_sample_prob)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/peacock.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    config = read_config(args.config)

    set_random_seed(config.seed)
    scene_aabb = torch.tensor(config.aabb, dtype=torch.float32, device=config.device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * np.sqrt(3) / config.n_samples
    ).item()

    log_dir = config.log
    assert os.path.exists(log_dir)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    os.makedirs(f"{log_dir}/eval/rgb", exist_ok=True)
    os.makedirs(f"{log_dir}/eval/depth", exist_ok=True)

    # load radiance field
    checkpoint = torch.load(f"{log_dir}/ckpt/ckpt.pth")

    estimator = OccGridEstimator(
        roi_aabb=config.aabb, resolution=config.grid_resolution, levels=config.grid_nlvl
    ).to(config.device)
    estimator.load_state_dict(checkpoint["estimator"])

    radiance_field = NGPradianceField(
        aabb=scene_aabb,
        density_activation=lambda x: F.softplus(x - 1),
        use_normal_net=config.use_normal_net,
        use_bkgd_net=config.use_bkgd_net,
        density_bias_scale=config.density_bias_scale,
        offset_scale=config.offset_scale,
    ).to(config.device)
    radiance_field.load_state_dict(checkpoint["radiance_field"])

    # setup dataset
    width, height = 512, 512
    test_dataset = DreamFusionLoader(
        size=36,
        width=width,
        height=height,
        training=False,
        device=config.device,
    )

    # evaluation
    radiance_field.eval()

    with torch.no_grad():
        for i in trange(len(test_dataset), desc="Evaluation"):
            data = test_dataset[i]
            rays = data["rays"]
            render_bkgd = data["color_bkgd"]
            shading = data["shading"]

            # rendering
            rgb, acc, depth, _, _ = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=config.near_plane,
                far_plane=config.far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=config.cone_angle,
                alpha_thre=config.alpha_thre,
                shading=shading,
                use_bkgd_net=False,
                chunk_size=config.eval_chunk_size,
            )

            rgb = rearrange(rgb, "(h w) c -> h w c", h=height, w=width)
            depth = rearrange(depth, "(h w) 1 -> h w", h=height, w=width)

            # save visualizations
            imageio.imwrite(
                f"{log_dir}/eval/rgb/{i}.png",
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                f"{log_dir}/eval/depth/{i}.png",
                ((depth / (depth.max() + 1e-6)).cpu().numpy() * 255).astype(np.uint8),
            )
