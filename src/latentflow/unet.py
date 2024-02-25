import logging
import torch

from .flow import Flow
from .latent import Latent, NoisePredict
from .prompt_embeddings import PromptEmbeddings
from .state import State
from .debug import tensor_hash
from .tensor import Tensor
from .controlnet import ControlNetLatent

class Unet(Flow):
    def __init__(self,
            unet,
            scheduler,
            fp16 = False,
            freeu = None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):

        self.unet = unet
        self.freeu = freeu
        self.scheduler = scheduler
        self.fp16 = fp16
        self.onload_device = onload_device
        self.offload_device = offload_device

        logging.debug("Unet init %s %s fp16=%s", unet.device, unet.dtype, self.fp16)
        logging.debug("Unet init %s", type(scheduler))

    def __call__(self,
            timestep,
            latent=None,
            embeddings=None,
            scheduler=None,
            ):

        self.timestep = timestep
        self.latent = latent
        self.embeddings = embeddings

        assert embeddings is not None

        if scheduler is not None:
            self.scheduler = scheduler

        logging.debug("Unet set %s %s %s",
                self.timestep, self.latent,
                self.embeddings,
                )

        return self

    def onload(self):
        self.unet = self.unet.to(self.onload_device)
        if self.freeu is not None:
            self.unet.enable_freeu(*self.freeu)

    def offload(self):
        self.unet = self.unet.to(self.offload_device)

    @torch.no_grad()
    def apply(self, latent):

        if self.latent is not None:
            latent = self.latent


        self.onload()
        latent.onload()
        self.embeddings.onload()


        timestep = self.timestep
        embeddings = self.embeddings

        down_block_res_samples = None
        mid_block_res_sample = None
        if isinstance(latent, ControlNetLatent):
            down_block_res_samples = latent.down_block_res_samples
            mid_block_res_sample = latent.mid_block_res_sample

        logging.debug("Unet apply %s %s %s",
                timestep, latent, embeddings,
                )

        logging.debug("Unet apply cnet %s %s",
                [len(down_block_res_samples), down_block_res_samples[0].dtype]
                    if down_block_res_samples is not None else None,

                [mid_block_res_sample.shape, mid_block_res_sample.dtype]
                    if mid_block_res_sample is not None else None,
                )

        latents = latent.latent
        latent_model_input = latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        embeddings = embeddings.embeddings.to(dtype=self.unet.dtype)

        with torch.autocast(self.onload_device, enabled=self.fp16, dtype=torch.float16):
            noise_pred = self.unet(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    ).sample.to(device=latents.device, dtype=latents.dtype)

        result = NoisePredict(latent=noise_pred)

        result.offload()
        self.embeddings.offload()
        latent.offload()
        self.offload()


        return result

class CFGPrepare(Flow):
    def __init__(self,
            do_classifier_free_guidance=True,
            guidance_scale=7.5,
            ):
        self.do_classifier_free_guidance=do_classifier_free_guidance
        self.guidance_scale=guidance_scale

    def apply(self, latent):
        latent_model_input = latent.latent
        latent_model_input = latent_model_input.repeat(2 if self.do_classifier_free_guidance else 1, 1, 1, 1, 1)
        return Latent(latent_model_input)

class CFGProcess(Flow):
    def __init__(self,
            do_classifier_free_guidance=True,
            guidance_scale=7.5,
            ):
        self.do_classifier_free_guidance=do_classifier_free_guidance
        self.guidance_scale=guidance_scale

    def apply(self, noise_predict):
        noise_pred =  noise_predict.latent
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        return NoisePredict(noise_pred)
