"""
Original code: https://github.com/ashawkey/stable-dreamfusion
"""

from diffusers import IFPipeline
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class IF(nn.Module):
    def __init__(self, device, vram_O=True, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = device

        print(f"[INFO] loading DeepFloyd IF-I-XL...")

        model_key = "DeepFloyd/IF-I-XL-v1.0"

        # Create model
        pipe = IFPipeline.from_pretrained(
            model_key, variant="fp16", torch_dtype=torch.float16
        )

        if vram_O:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.unet = pipe.unet
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        self.pipe = pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f"[INFO] loaded DeepFloyd IF-I-XL!")

    @torch.no_grad()
    def compute_text_emb(self, prompt, negative_prompt=""):
        # prompt: [str]

        text_emb, uncond_emb = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device=self.device
        )
        text_emb = torch.cat([uncond_emb, text_emb])
        return text_emb

    def sds(self, text_emb, rgb, guidance_scale=20, grad_scale=1):
        num_rays, _ = rgb.shape
        h = w = int(num_rays ** (1 / 2))
        rgb = rearrange(rgb, "(h w) c -> 1 c h w", h=h, w=w)

        # [0, 1] to [-1, 1] and make sure shape is [64, 64]
        images = (
            F.interpolate(rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(images)
            images_noisy = self.scheduler.add_noise(images, noise, t)

            # pred noise
            model_input = torch.cat([images_noisy] * 2)
            model_input = self.scheduler.scale_model_input(model_input, t)
            noise_pred = self.unet(
                model_input, t, encoder_hidden_states=text_emb
            ).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(
                model_input.shape[1], dim=1
            )
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(images, grad)

        return loss
