import torch
import logging
import PIL
import numpy as np
from einops import rearrange
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.models import AutoencoderKL, ControlNetModel
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent

class ControlNetLatent(Latent):
    def __init__(self,
            latent,
            down_block_res_samples,
            mid_block_res_sample,
            onload_device='cuda',
            offload_device='cpu',
            ):
        super().__init__(
                latent=latent.latent,
                onload_device=onload_device,
                offload_device=offload_device,
                )

        self.down_block_res_samples = down_block_res_samples
        self.mid_block_res_sample = mid_block_res_sample

    def onload(self):
        super().onload()

        if self.down_block_res_samples is not None:
            for i, _ in enumerate(self.down_block_res_samples):
                self.down_block_res_samples[i] = self.down_block_res_samples[i].to(self.onload_device)

        if self.mid_block_res_sample is not None:
            self.mid_block_res_sample = self.mid_block_res_sample.to(self.onload_device)

    def offload(self):
        super().offload()

        if self.down_block_res_samples is not None:
            for i, _ in enumerate(self.down_block_res_samples):
                self.down_block_res_samples[i] = self.down_block_res_samples[i].to(self.offload_device)

        if self.mid_block_res_sample is not None:
            self.mid_block_res_sample = self.mid_block_res_sample.to(self.offload_device)


class ControlNet(Flow):
    def __init__(self,
            controlnet = None,
            controlnet_path = None,
            scheduler = None,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            guess_mode: bool = False,
            do_classifier_free_guidance: bool = True,
            dtype=torch.float32,
            onload_device='cuda',
            offload_device='cpu',
            ):

        self.onload_device = onload_device
        self.offload_device = offload_device
        self.dtype = dtype

        if controlnet_path is not None:
            controlnet = MultiControlNetModel(ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype))

        if isinstance(controlnet, ControlNetModel):
            controlnet = MultiControlNetModel([controlnet])

        assert isinstance(controlnet, MultiControlNetModel)

        self.controlnet = controlnet
        self.scheduler = scheduler
        self.guess_mode = guess_mode

        self.control_guidance_start = control_guidance_start
        self.control_guidance_end = control_guidance_end

        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            self.control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            self.control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            self.control_guidance_start, self.control_guidance_end = mult * [self.control_guidance_start], mult * [
                self.control_guidance_end
            ]

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        self.guess_mode = self.guess_mode or global_pool_conditions

    def onload(self):
        self.controlnet = self.controlnet.to(self.onload_device)

    def offload(self):
        self.controlnet = self.controlnet.to(self.offload_device)

    def set(self, controlnet_video):
        self.controlnet_images = controlnet_video.chw().float()/255.0

    def __call__(self,
            timestep_index,
            timestep,
            latent=None,
            image=None,
            timesteps=None,
            controlnet_scale = 1.0,
            embeddings = None,
            ):

        self.timestep_index = timestep_index
        self.timestep = timestep
        self.timesteps = timesteps
        self.latent = latent
        self.embeddings = embeddings

        if image is not None:
            self.controlnet_images = image

        self.controlnet_scale = controlnet_scale \
                if isinstance(controlnet_scale, list) \
                else [controlnet_scale] * len(self.controlnet.nets)

        return self

    @torch.no_grad()
    def apply(self, latent):

        if self.latent is not None:
            latent = self.latent

        self.onload()
        latent.onload()
        self.timesteps.onload()
        self.embeddings.onload()

        timestep_index = self.timestep_index
        timestep = self.timestep
        timesteps = self.timesteps

        embeddings = self.embeddings.embeddings.to(self.controlnet.dtype)

        latent_model_input = latent.latent.to(self.controlnet.device, self.controlnet.dtype)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        controlnet_images = self.controlnet_images
        if not isinstance(controlnet_images, list):
            controlnet_images = [controlnet_images]

        for i, image in enumerate(controlnet_images):
            controlnet_images[i] = controlnet_images[i].squeeze(0)
            controlnet_images[i] = controlnet_images[i].to(
                device=self.controlnet.device,
                dtype=self.controlnet.dtype)

            logging.debug("ControlNet images %s %s %s", i, controlnet_images[i].shape, controlnet_images[i].dtype)

        logging.debug("ControlNet embeddings %s %s", embeddings.shape, embeddings.dtype)
        logging.debug("ControlNet latents %s %s", latent_model_input.shape, latent_model_input.dtype)

        controlnet_conditioning_scale = self.controlnet_scale

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        temporal_context = latent_model_input.shape[2]

        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(self.control_guidance_start, self.control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps)

        down_block_res_samples, mid_block_res_sample = self.calc_cnet_residuals(
                timestep_index,
                timestep,
                embeddings,
                controlnet_images,
                latent_model_input,
                controlnet_keep,
                controlnet_conditioning_scale,
                self.guess_mode,
                temporal_context,
                self.do_classifier_free_guidance,
                )

        if isinstance(latent, ControlNetLatent):
            down_block_res_samples = [
                samples_prev + samples_curr
                for samples_prev, samples_curr in zip(down_block_res_samples, latent.down_block_res_samples)
            ]
            mid_block_res_sample += latent.mid_block_res_sample

        result = ControlNetLatent(latent, down_block_res_samples, mid_block_res_sample)

        result.offload()
        self.embeddings.offload()
        self.timesteps.offload()
        latent.offload()
        self.offload()

        return result

    @torch.no_grad()
    def calc_cnet_residuals(
            self,
            step_index,
            timestep,
            embeddings,
            controlnet_image,
            latent_model_input,
            controlnet_keep,
            controlnet_conditioning_scale,
            guess_mode,
            temporal_context,
            do_classifier_free_guidance):

        down_block_res_samples = None
        mid_block_res_sample = None

        # controlnet(s) inference
        if guess_mode and do_classifier_free_guidance:
            # Infer ControlNet only for the conditional batch.
            control_model_input = latent_model_input
            control_model_input = self.scheduler.scale_model_input(control_model_input, timestep)
        else:
            control_model_input = latent_model_input

        if isinstance(controlnet_keep[step_index], list):
            cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_index])]
        else:
            controlnet_cond_scale = controlnet_conditioning_scale
            if isinstance(controlnet_cond_scale, list):
                controlnet_cond_scale = controlnet_cond_scale[0]
            cond_scale = controlnet_cond_scale * controlnet_keep[step_index]

        if do_classifier_free_guidance and not guess_mode:
            for i, _ in enumerate(controlnet_image):
                controlnet_image[i] = controlnet_image[i].repeat(2,1,1,1)

        control_model_input = rearrange(control_model_input, "b c f h w -> (b f) c h w")

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            control_model_input,
            timestep,
            encoder_hidden_states=embeddings,
            controlnet_cond=controlnet_image,
            conditioning_scale=cond_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        for down_idx in range(len(down_block_res_samples)):
            down_block_res_samples[down_idx] = rearrange(
                    down_block_res_samples[down_idx],
                    '(b f) c h w -> b c f h w',
                    f=temporal_context)

        mid_block_res_sample = rearrange(
                mid_block_res_sample,
                '(b f) c h w -> b c f h w',
                f=temporal_context)

        if guess_mode and do_classifier_free_guidance:
            # Infered ControlNet only for the conditional batch.
            # To apply the output of ControlNet to both the unconditional and conditional batches,
            # add 0 to the unconditional batch to keep it unchanged.
            down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
            mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

        return down_block_res_samples, mid_block_res_sample

