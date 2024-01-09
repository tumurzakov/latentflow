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
            ):

        self.unet = unet
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = do_classifier_free_guidance

        logging.debug("Unet init %s %s", unet.device, unet.dtype)
        logging.debug("Unet init %s %s", guidance_scale, do_classifier_free_guidance)
        logging.debug("Unet init %s", type(scheduler))

    @torch.no_grad()
    def apply(self, state: State) -> Latent:

        timestep = state['timestep']
        latent = state['latent']
        embeddings = state['embeddings']

        logging.debug("Unet %s %s %s",
                timestep, latent, embeddings)

        latents = latent.latent
        latent_model_input = latents.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        prompt_embeddings = embeddings.embeddings

        noise_pred = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeddings).sample

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

