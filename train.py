import argparse
import os
import shutil
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from nerfacc.estimators.occ_grid import OccGridEstimator
from tqdm import tqdm, trange

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
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    os.makedirs(f"{log_dir}/ckpt")
    os.makedirs(f"{log_dir}/rgb")
    os.makedirs(f"{log_dir}/depth")

    # setup radiance field
    estimator = OccGridEstimator(
        roi_aabb=config.aabb, resolution=config.grid_resolution, levels=config.grid_nlvl
    ).to(config.device)

    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPradianceField(
        aabb=scene_aabb,
        density_activation=lambda x: F.softplus(x - 1),
        use_normal_net=config.use_normal_net,
        use_bkgd_net=config.use_bkgd_net,
        density_bias_scale=config.density_bias_scale,
        offset_scale=config.offset_scale,
    ).to(config.device)

    optimizer = torch.optim.Adam(
        radiance_field.get_params(lr=config.lr),
        lr=config.lr,
        betas=(0.9, 0.99),
        eps=1e-15,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

    # setup guidance model
    if config.guidance == "if":
        from guidance.deepfloyd_if import IF

        guidance = IF(device=config.device)
    elif config.guidance == "stable-diffusion":
        from guidance.stable_diffusion import StableDiffusion

        guidance = StableDiffusion(device=config.device, sd_version=config.sd_version)
    for p in guidance.parameters():
        p.requires_grad = False

    # setup dataset
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

    # prepare text embeddings
    text_embs = {
        direction: guidance.compute_text_emb(f"{config.text}, {direction} view")
        for direction in ["front", "side", "back", "side", "top", "bottom"]
    }

    # training
    tic = time.time()
    for step in trange(config.max_steps + 1, desc=f"Step"):
        radiance_field.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        data = train_dataset[i]
        rays = data["rays"]
        direciton = data["direction"]

        if step < config.max_steps * 0.2:
            render_bkgd = torch.ones_like(data["color_bkgd"])
            shading = "albedo"
        else:
            render_bkgd = data["color_bkgd"]
            shading = data["shading"] if config.use_shading else "albdeo"
        text_emb = text_embs[direciton]

        def occ_eval_fn(x):
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        estimator.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
            n=config.grid_update_interval,
            ema_decay=config.grid_update_ema_decay,
        )

        # render
        rgb, acc, depth, loss_orient, n_rendering_samples = render_image_with_occgrid(
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
            use_bkgd_net=config.use_bkgd_net,
        )
        if n_rendering_samples == 0:
            continue

        # compute loss
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss = guidance.sds(
                text_emb,
                rgb,
                guidance_scale=config.guidance_scale,
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

        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        # logging
        if step % config.log_interval == 0 or step == config.max_steps - 1:
            elapsed_time = time.time() - tic
            tqdm.write(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss.item():.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | max_depth={depth.max():.3f} | "
            )

        # save checkpoint
        if step % config.save_interval == 0 or step == config.max_steps - 1:
            save_dict = {
                "estimator": estimator.state_dict(),
                "radiance_field": radiance_field.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, f"{log_dir}/ckpt/ckpt.pth")
            tqdm.write(f"[INFO] Saved checkpoint at {log_dir}/ckpt/ckpt.pth")

        # evaluation
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            radiance_field.eval()

            with torch.no_grad():
                for j in trange(len(test_dataset), desc="Eval"):
                    data = test_dataset[j]
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

                    rgb = rearrange(
                        rgb, "(h w) c -> h w c", h=config.eval_h, w=config.eval_w
                    )
                    depth = rearrange(
                        depth, "(h w) 1 -> h w", h=config.eval_h, w=config.eval_w
                    )

                    # save visualizations
                    imageio.imwrite(
                        f"{log_dir}/rgb/{j}_step{step}.png",
                        (rgb.cpu().numpy() * 255).astype(np.uint8),
                    )
                    imageio.imwrite(
                        f"{log_dir}/depth/{j}_step{step}.png",
                        ((depth / (depth.max() + 1e-6)).cpu().numpy() * 255).astype(
                            np.uint8
                        ),
                    )
