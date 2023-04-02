import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd
from transformers import CLIPTextModel, CLIPTokenizer, logging


# suppress partial model loading warning
logging.set_verbosity_error()


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
        return gt_grad * grad_scale, None


class StableDiffusion(nn.Module):
    def __init__(self, sd_version="2.1", concept_name=None, hf_key=None):
        super().__init__()
        self.sd_version = sd_version

        print(f"[INFO] loading stable diffusion...")

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        elif self.sd_version == "1.4":
            model_key = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_key, subfolder="text_encoder"
        )
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet")

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod  # for convenience

        print(f"[INFO] loaded stable diffusion!")

        if concept_name:
            self.load_concept(concept_name)

    def compute_text_emb(self, prompt, negative_prompt="", direction=""):
        if direction:
            prompt = f"{prompt}, {direction} view."

        # prompt, negative_prompt: [str]
        prompt, negative_prompt, direction = [prompt], [negative_prompt], [direction]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            text_emb = self.text_encoder(
                text_input.input_ids.to(self.text_encoder.device)
            )[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            uncond_emb = self.text_encoder(
                uncond_input.input_ids.to(self.text_encoder.device)
            )[0]

        # Cat for final embeddings
        text_emb = torch.cat([uncond_emb, text_emb])
        return text_emb

    def encode_rgb(self, rgb):
        # rgb: [B, 3, H, W]
        rgb = 2 * rgb - 1

        posterior = self.vae.encode(rgb).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            rgb = self.vae.decode(latents).sample

        rgb = (rgb / 2 + 0.5).clamp(0, 1)

        return rgb

    def sds(self, text_emb, rgb, guidance_scale=100):
        """Score distillation sampling"""

        if len(rgb.shape) == 2:
            num_rays, _ = rgb.shape
            h = w = int(num_rays ** (1 / 2))
            rgb = rearrange(rgb, "(h w) c -> 1 c h w", h=h, w=w)
        else:
            rgb = rearrange(rgb, "h w c -> 1 c h w")

        rgb = F.interpolate(rgb, (512, 512), mode="bilinear", align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [1],
            dtype=torch.long,
            device=self.unet.device,
        )

        # encode image into latents with vae, requires grad!
        latents = self.encode_rgb(rgb)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_emb
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.alphas.device is not self.unet.device:
            self.alphas = self.alphas.to(self.unet.device)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss
