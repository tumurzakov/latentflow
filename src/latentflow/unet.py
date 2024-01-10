import logging
import torch

from .flow import Flow
from .latent import Latent
from .prompt_embeddings import PromptEmbeddings
from .state import State

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

    def __call__(self, timestep_index, timestep, controlnet_scale = 1.0):

        self.timestep_index = timestep_index
        self.timestep = timestep

        return self


    @torch.no_grad()
    def apply(self, state: State) -> Latent:

        timestep = self.timestep

        latent = state['latent']
        embeddings = state['embeddings']
        down_block_res_samples = state['down_block_res_samples']
        mid_block_res_sample = state['mid_block_res_sample']

        logging.debug("Unet %s %s %s",
                timestep, latent, embeddings)

        latents = latent.latent
        latent_model_input = latents.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        prompt_embeddings = embeddings.embeddings

        with torch.autocast('cuda', enabled=self.fp16, dtype=torch.float16):
            noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    ).sample

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latent.latent = self.scheduler.step(noise_pred,
                timestep,
                latents,
                ).prev_sample

        return latent

