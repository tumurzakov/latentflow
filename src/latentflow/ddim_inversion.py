import os
import imageio
import numpy as np
import hashlib
from typing import Union

import torch
import torchvision

from tqdm import tqdm

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, tokenizer, text_encoder):
    uncond_input = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(text_encoder.device))[0]
    text_input = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = text_encoder(text_input.input_ids.to(text_encoder.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    timestep = timestep.to(torch.int)
    next_timestep = next_timestep.to(torch.int)
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.alphas_cumprod[0]
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(tokenizer, text_encoder, unet, scheduler, latent, num_inv_steps, prompt, desc=""):
    context = init_prompt(prompt, tokenizer, text_encoder)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps), desc=desc):
        t = scheduler.timesteps[len(scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, unet)
        latent = next_step(noise_pred, t, latent, scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(tokenizer, text_encoder, unet, scheduler, video_latent, num_inv_steps, prompt="", desc=""):
    ddim_latents = ddim_loop(tokenizer, text_encoder, unet, scheduler, video_latent, num_inv_steps, prompt, desc)
    return ddim_latents
