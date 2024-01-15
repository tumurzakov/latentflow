import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings
from .mask import Mask

class Region(Flow):
    def __init__(self,
            controlnet_scale: List = None,
            mask: Mask=None,
            loras = None,
            guidance_scale = None,
            ):
        self.controlnet_scale = controlnet_scale
        self.mask = mask
        self.loras = loras
        self.guidance_scale = guidance_scale

    def apply(self, embeddings: PromptEmbeddings):
        self.embeddings = embeddings

        return self

    def __str__(self):
        return 'Region(%s, %s, %s)' % (
                self.controlnet_scale,
                self.mask,
                self.embeddings,
                )


