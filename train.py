import argparse
import os
import shutil
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from attrdict import AttrDict
from einops import rearrange
from nerfacc import ContractionType, OccupancyGrid
from tqdm import tqdm, trange

from dataset.dreamfusion import DreamFusionLoader
from prior.diffusion import StableDiffusion
from utils import render_image, set_random_seed


def read_config(fn: str = None):
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
        default="config/magic3d.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    config = read_config(args.config)

    set_random_seed(config.seed)
    scene_aabb = torch.tensor(config.aabb, dtype=torch.float32, device=config.device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * np.sqrt(3) / config.n_samples
    ).item()

    if config.backbone == "mlp":
        from model.mlp import VanillaNeRFRadianceField

        grad_scaler = torch.cuda.amp.GradScaler(1)
        radiance_field = VanillaNeRFRadianceField().to(config.device)
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
    elif config.backbone == "ngp":
        from model.ngp import NGPradianceField

        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field = NGPradianceField(
            aabb=scene_aabb,
            density_activation=lambda x: F.softplus(x - 1),
            use_predict_normal=config.use_predict_normal,
            density_bias_scale=config.density_bias_scale,
            offset_scale=config.offset_scale,
            use_predict_bkgd=config.use_predict_bkgd,
        ).to(config.device)
        optimizer = torch.optim.Adam(
            radiance_field.get_params(lr=config.lr), lr=config.lr, betas=(0.9, 0.99), eps=1e-15
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    occupancy_grid = OccupancyGrid(
        roi_aabb=config.aabb,
        resolution=config.grid_resolution,
        contraction_type=ContractionType.AABB,
    ).to(config.device)

    # setup stable-diffusion as guidance model
    guidance = StableDiffusion(sd_version=config.sd_version).to(config.device)
    for p in guidance.parameters():
        p.requires_grad = False

    log_dir = config.log
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    os.makedirs(f"{log_dir}/rgb")
    os.makedirs(f"{log_dir}/acc")
    os.makedirs(f"{log_dir}/depth")

    train_dataset = DreamFusionLoader(
        size=config.train_dataset_size,
        width=config.train_w,
        height=config.train_h,
        shading_sample_prob=config.shading_sample_prob
        if config.use_shading
        else [1, 0, 0],
        device=config.device,
    )
    test_dataset = DreamFusionLoader(
        size=config.eval_dataset_size,
        width=config.eval_w,
        height=config.eval_h,
        training=False,
        device=config.device,
    )
    text_embs = {
        direction: guidance.compute_text_emb(config.text, direction=direction)
        for direction in ["front", "side", "back", "side", "top", "bottom"]
    }

    # * training
    tic = time.time()
    for step in trange(config.max_steps + 1, desc=f"Step"):
        radiance_field.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        radiance_field.train()

        data = train_dataset[i]["novel"]
        rays = data["rays"]
        direciton = data["direction"]
        render_bkgd = data["color_bkgd"]
        shading = (
            data["shading"]
            if step > config.start_shading_step and config.use_shading
            else "albedo"
        )
        text_emb = text_embs[direciton]

        def occ_eval_fn(x):
            # compute occupancy
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        occupancy_grid.every_n_step(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
            n=config.grid_update_interval,
            ema_decay=config.grid_update_ema_decay,
        )

        # render
        rgb, acc, depth, loss_orient, n_rendering_samples = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=config.near_plane,
            far_plane=config.far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=config.cone_angle,
            alpha_thre=config.alpha_thre,
            shading=shading,
            use_predict_bkgd=config.use_predict_bkgd,
        )
        if n_rendering_samples == 0:
            continue

        # compute loss
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss = guidance.sds(
                text_emb,
                rgb,
                guidance_scale=100,
            )

            if config.use_orient_loss and loss_orient is not None:
                loss += config.lambda_orient * loss_orient
            if config.use_opacity_loss:
                loss += config.lambda_opacity * (((acc**2) + 0.01) ** (1 / 2)).mean()
            if config.use_entropy_loss:
                alphas = acc.clamp(1e-5, 1 - 1e-5)
                loss_entropy = (
                    config.lambda_entropy
                    * (
                        -alphas * torch.log2(alphas)
                        - (1 - alphas) * torch.log2(1 - alphas)
                    ).mean()
                )

        # ! do not unscale it because we are using Adam
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        # * logging
        if step % config.log_interval == 0 or step == config.max_steps:
            elapsed_time = time.time() - tic
            tqdm.write(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss.item():.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | max_depth={depth.max():.3f} | "
            )

        # * evaluation
        if step % config.eval_interval == 0 or step == config.max_steps:
            radiance_field.eval()

            with torch.no_grad():
                for j in trange(len(test_dataset), desc="Evaluation"):
                    data = test_dataset[j]["novel"]
                    rays = data["rays"]
                    render_bkgd = data["color_bkgd"]
                    shading = data["shading"]

                    # rendering
                    rgb, acc, depth, _, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb,
                        # rendering options
                        near_plane=config.near_plane,
                        far_plane=config.far_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=config.cone_angle,
                        alpha_thre=config.alpha_thre,
                        shading=shading,
                        use_predict_bkgd=config.use_predict_bkgd,
                        # test options
                        eval_chunk_size=config.eval_chunk_size,
                    )

                    if len(rgb.shape) == 2:
                        rgb = rearrange(
                            rgb, "(h w) c -> h w c", h=config.eval_h, w=config.eval_w
                        )
                        acc = rearrange(
                            acc, "(h w) c -> h w c", h=config.eval_h, w=config.eval_w
                        )
                        depth = rearrange(
                            depth, "(h w) 1 -> h w", h=config.eval_h, w=config.eval_w
                        )

                    # render visualizations
                    imageio.imwrite(
                        f"{log_dir}/acc/{j}_step{step}.png",
                        ((acc > 0).float().cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        f"{log_dir}/rgb/{j}_step{step}.png",
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        f"{log_dir}/depth/{j}_step{step}.png",
                        (depth.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                    )
