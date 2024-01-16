import logging
import torch

from .flow import Flow
from .latent import Latent, NoisePredict
from .prompt_embeddings import PromptEmbeddings
from .state import State
from .debug import tensor_hash

class Unet(Flow):
    def __init__(self,
            unet,
            scheduler,
            guidance_scale=7.5,
            do_classifier_free_guidance = True,
            fp16 = False,
            ):

        self.unet = unet
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.fp16 = fp16

        logging.debug("Unet init %s %s fp16=%s", unet.device, unet.dtype, self.fp16)
        logging.debug("Unet init %s %s", guidance_scale, do_classifier_free_guidance)
        logging.debug("Unet init %s", type(scheduler))

    def __call__(self,
            timestep,
            latent,
            embeddings,
            guidance_scale=None,
            scheduler=None,
            ):

        self.timestep = timestep
        self.latent = latent
        self.embeddings = embeddings

        if scheduler is not None:
            self.scheduler = scheduler

        if guidance_scale is not None:
            self.guidance_scale = guidance_scale

        logging.debug("Unet set %s %s %s %s",
                self.timestep, self.latent,
                self.embeddings, self.guidance_scale)

        return self


    @torch.no_grad()
    def apply(self, state: State):

        timestep = self.timestep
        latent = self.latent
        embeddings = self.embeddings

        down_block_res_samples = state['down_block_res_samples']
        mid_block_res_sample = state['mid_block_res_sample']

        logging.debug("Unet apply %s %s %s %s",
                timestep, latent, embeddings, self.guidance_scale,
                )

        logging.debug("Unet apply cnet %s %s",
                len(down_block_res_samples) if down_block_res_samples is not None else None,
                mid_block_res_sample.shape if mid_block_res_sample is not None else None,
                )

        latents = latent.latent
        latent_model_input = latents
        latent_model_input = latent_model_input.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        embeddings = embeddings.embeddings.to(self.unet.device, dtype=self.unet.dtype)

        with torch.autocast('cuda', enabled=self.fp16, dtype=torch.float16):
            noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    ).sample.to(device=latents.device, dtype=latents.dtype)

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return NoisePredict(latent=noise_pred)
