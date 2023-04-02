import argparse
import os
import shutil
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from nerfacc import ContractionType, OccupancyGrid
from tqdm import tqdm, trange

from dataset.dreamfusion import DreamFusionLoader
from model.ngp import NGPradianceField
from prior.diffusion import StableDiffusion
from utils import render_image, set_random_seed


if __name__ == "__main__":

    device = "cuda:5"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        default="A DSLR photo of a peacock",
        help="text prompt",
    )
    parser.add_argument(
        "--image_size",
        type=lambda s: [int(item) for item in s.split("x")],
        default="256x256",
        help="image size in wxh, e.g, 128x128",
    )
    parser.add_argument(
        "--log",
        type=str,
        # default="log/dream/test",
        # default="log/dream/sand_castle_3_with_shading_orient",
        # default="log/dream/hamburger_bg_random",
        # default="log/dream/test-lr-1e2-orient",
        default="log/dream/test-lr-1e2-bound1",
        help="which folder to log",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--test_chunk_size",
        type=int,
        # default=4096,  # cutlass_matmul.h:332 error occurs
        default=torch.iinfo(torch.int32).max,
    )
    parser.add_argument(
        "--aabb",
        type=lambda s: [float(item) for item in s.split(",")],
        # default="-2.0,-2.0,-2.0,2.0,2.0,2.0",
        default="-1.0,-1.0,-1.0,1.0,1.0,1.0",
        help="delimited list input",
    )
    parser.add_argument("--cone_angle", type=float, default=0.000)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.0-depth", "2.1"],
        help="which version of stable diffusion",
    )
    args = parser.parse_args()

    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
    near_plane = 0.1
    far_plane = None

    # bounded setting
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max() * np.sqrt(3) / 512
    ).item()
    alpha_thre = 0.0
    cone_angle = 0.0

    # setup the radiance field
    grid_resolution = 128
    max_steps = args.max_steps

    # regularization
    use_orient_loss = True
    # use_orient_loss = False
    lambda_orient = 1e-2
    use_opacity_loss = False
    lambda_opacity = 1e-3
    use_entropy_loss = False
    lambda_entropy = 1e-4

    grad_scaler = torch.cuda.amp.GradScaler()
    radiance_field = NGPradianceField(
        aabb=scene_aabb, density_activation=lambda x: F.softplus(x - 1)
    ).to(device)
    optimizer = torch.optim.Adam(
        radiance_field.parameters(),
        lr=1e-2,
        betas=(0.9, 0.99),
        eps=1e-15
        # radiance_field.parameters(), lr=1e-3, betas=(0.9, 0.99), eps=1e-15
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer,
    #     milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
    #     gamma=0.33,
    # )

    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=grid_resolution,
        contraction_type=contraction_type,
    ).to(device)

    # setup stable-diffusion as guidance model
    guidance = StableDiffusion(sd_version=args.sd_version).to(device)
    for p in guidance.parameters():
        p.requires_grad = False

    log_dir = args.log
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    os.makedirs(f"{log_dir}/rgb")
    os.makedirs(f"{log_dir}/acc")
    os.makedirs(f"{log_dir}/depth")
    os.makedirs(f"{log_dir}/normal")

    width, height = args.image_size
    # width_eval, height_eval = 256, 256
    width_eval, height_eval = 800, 800
    train_dataset = DreamFusionLoader(
        size=100, width=width, height=height, device=device
    )
    test_dataset = DreamFusionLoader(
        size=8, width=width_eval, height=height_eval, training=False, device=device
    )
    text_embs = {
        direction: guidance.compute_text_emb(args.text, direction=direction)
        for direction in ["front", "side", "back", "side", "top", "bottom"]
    }

    # training
    tic = time.time()
    for step in trange(max_steps + 1, desc=f"Step"):
        radiance_field.train()

        i = torch.randint(0, len(train_dataset), (1,)).item()
        radiance_field.train()

        data = train_dataset[i]["novel"]
        rays = data["rays"]
        direciton = data["direction"]
        render_bkgd = data["color_bkgd"]
        shading = data["shading"] if step > 1000 else "albedo"
        text_emb = text_embs[direciton]

        def occ_eval_fn(x):
            # compute occupancy
            density = radiance_field.query_density(x)
            return density * render_step_size

        # update occupancy grid
        occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2, n=10, ema_decay=0.6)

        # render
        rgb, acc, depth, normal, loss_orient, n_rendering_samples = render_image(
            radiance_field,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre,
            shading=shading,
        )
        if n_rendering_samples == 0:
            continue

        # compute loss
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            loss = guidance.sds(
                text_emb,
                rgb,
                depth=depth if args.sd_version == "2.0-depth" else None,
                guidance_scale=100,
            )

            if use_orient_loss:
                loss += lambda_orient * loss_orient
            if use_opacity_loss:
                loss += lambda_opacity * (((acc**2) + 0.01) ** (1 / 2)).mean()
            if use_entropy_loss:
                alphas = acc.clamp(1e-5, 1 - 1e-5)
                loss_entropy = (
                    lambda_entropy
                    * (
                        -alphas * torch.log2(alphas)
                        - (1 - alphas) * torch.log2(1 - alphas)
                    ).mean()
                )

        # do not unscale it because we are using Adam.
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()

        # * logging
        if step % 100 == 0 or step == max_steps:
            elapsed_time = time.time() - tic
            tqdm.write(
                f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                f"loss={loss.item():.2f} | "
                f"n_rendering_samples={n_rendering_samples:d} | max_depth={depth.max():.3f} | "
                f"num_rays={height*width:d} | "
            )

        # * evaluation
        if step % 100 == 0 or step == max_steps:
            radiance_field.eval()

            with torch.no_grad():
                for j in trange(len(test_dataset), desc="Evaluation"):
                    data = test_dataset[j]["novel"]
                    rays = data["rays"]
                    render_bkgd = data["color_bkgd"]
                    shading = data["shading"]

                    # rendering
                    rgb, acc, depth, normal, _, _ = render_image(
                        radiance_field,
                        occupancy_grid,
                        rays,
                        scene_aabb,
                        # rendering options
                        near_plane=near_plane,
                        far_plane=far_plane,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        cone_angle=cone_angle,
                        alpha_thre=alpha_thre,
                        shading=shading,
                    )

                    if len(rgb.shape) == 2:
                        rgb = rearrange(
                            rgb, "(h w) c -> h w c", h=height_eval, w=width_eval
                        )
                        acc = rearrange(
                            acc, "(h w) c -> h w c", h=height_eval, w=width_eval
                        )
                        depth = rearrange(
                            depth, "(h w) 1 -> h w", h=height_eval, w=width_eval
                        )
                        normal = rearrange(
                            normal, "(h w) c -> h w c", h=height_eval, w=width_eval
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
                    imageio.imwrite(
                        f"{log_dir}/normal/{j}_step{step}.png",
                        (normal.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                    )
