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
            controlnet_video: List = None,
            controlnet_scale: List = None,
            mask: Mask=None,
            loras = {},
            guidance_scale = None,
            source_latent = None,
            latent = None,
            noise_predict = None,
            scheduler = None,
            name = None,
            start_timestep = None,
            stop_timestep = None,
            shrink = None,
            pixel_infer_count = None,
            latent_shrinked = None,
            ):
        self.scheduler = scheduler
        self.controlnet_video = controlnet_video
        self.controlnet_scale = controlnet_scale
        self.mask = mask
        self.loras = loras
        self.guidance_scale = guidance_scale
        self.source_latent = source_latent
        self.latent = latent
        self.noise_predict = noise_predict
        self.timesteps = Timesteps()
        self.name = name
        self.start_timestep = start_timestep
        self.stop_timestep = stop_timestep
        self.shrink = shrink
        self.pixel_infer_count = pixel_infer_count
        self.latent_shrinked = latent_shrinked

        self.key = None

    def apply(self, prompt: Prompt):
        self.prompt = prompt

        return self

    def __setitem__(self, key, value):
        self.__dict__.update({key: value})

    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        return None

    def __str__(self):
        return 'Region(%s, %s, %s)' % (
                self.controlnet_scale,
                self.mask,
                self.prompt,
                )
