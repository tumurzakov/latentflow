import logging
import torch
from typing import List, Optional, Tuple, Union, Generator

from .flow import Flow

class Prompt(Flow):
    def __init__(self,
            prompt: str = "",
            negative_prompt: str = "",
            image = None,
            negative_image = None,
            frames: List = None,
            embeddings = None,
            ):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.image = image
        self.negative_image = negative_image
        self.frames = frames
        self.prompts = None
        self.embeddings = embeddings

        if frames is not None:
            self.prompts = [None for x in range(max(frames)+1)]
            for x in range(max(frames)+1):
                if x in self.frames:
                    self.prompts[x] = Prompt(
                            prompt=self.prompt,
                            negative_prompt=self.negative_prompt,
                            image=self.image[:, x:x+1] if self.image is not None else None,
                            negative_image=self.negative_image[:, x:x+1] if self.negative_image is not None else None,
                            embeddings=self.embeddings,
                            )

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

    def save(self, path):
        assert self.embeddings is not None, "Embeddings are empty"
        self.embeddings.save(path)
        return self

    def load(self, path):
        self.embeddings = PromptEmbeddings().load(path)
        return self
