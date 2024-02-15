import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .latent import Latent, NoisePredict
from .flow import Flow
from .prompt import Prompt
from .mask import Mask
from .video import Video
from .timesteps import Timesteps

class Region(Flow):
    def __init__(self,
            controlnet_image: Video = None,
            controlnet_scale: List = None,
            mask: Mask=None,
            loras = None,
            guidance_scale = None,
            source_latent = None,
            latent = None,
            scheduler = None,
            ):
        self.scheduler = scheduler
        self.controlnet_image = controlnet_image
        self.controlnet_scale = controlnet_scale
        self.mask = mask
        self.loras = loras
        self.guidance_scale = guidance_scale
        self.source_latent = source_latent
        self.latent = latent
        self.timesteps = Timesteps()

    def apply(self, prompt: Prompt):
        self.prompt = prompt

        return self

    def __str__(self):
        return 'Region(%s, %s, %s)' % (
                self.controlnet_scale,
                self.mask,
                self.prompt,
                )
