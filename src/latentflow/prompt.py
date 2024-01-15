import logging
import torch
from typing import List, Optional, Tuple, Union, Generator

from .flow import Flow

class Prompt(Flow):
    def __init__(self,
            prompt: str = "",
            negative_prompt: str = "",
            frames: List = None,
            ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.frames = frames
        self.prompts = None

        if frames is not None:
            self.prompts = [None for x in range(max(frames)+1)]
            for x in range(max(frames)+1):
                if x in self.frames:
                    self.prompts[x] = Prompt(self.prompt, self.negative_prompt)

        if self.frames is not None:
            logging.debug(f"init {self}")

    def set(self, prompt):
        frames = [None for x in range(max(max(self.frames), max(prompt.frames))+1)]

        for i, f in enumerate(self.frames):
            if f is not None:
                frames[i] = f

        for i, f in enumerate(prompt.frames):
            if f is not None:
                frames[i] = f

        self.frames = frames

        return self

    def apply(self, other):
        if not isinstance(other, Prompt):
            return other

        assert other.frames is not None

        other.set(self)

        return other

    def __str__(self):
        return f'Prompt(+[{self.prompt}], -[{self.negative_prompt}])'
