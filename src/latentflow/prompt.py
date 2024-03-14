import logging
import torch
from typing import List, Optional, Tuple, Union, Generator

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings

class Prompt(Flow):
    def __init__(self,
            prompt: str = "",
            negative_prompt: str = "",
            image = None,
            negative_image = None,
            frames: List = None,
            prompts: List = None,
            embeddings = None,
            ):

        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.image = image
        self.negative_image = negative_image
        self.frames = frames
        self.prompts = prompts
        self.embeddings = embeddings

        if prompts is None and frames is not None:
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

    def onload(self):
        if self.image is not None:
            self.image.onload()

        if self.embeddings is not None:
            self.embeddings.onload()

    def offload(self):
        if self.image is not None:
            self.image.offload()

        if self.embeddings is not None:
            self.embeddings.offload()

    def __add__(self, prompt):
        assert prompt.prompts is not None
        assert prompt.frames is not None

        prompts = [None for x in range(max(max(self.frames),max(prompt.frames))+1)]
        frames = []

        for i, p in enumerate(self.prompts):
            if p is not None:
                prompts[i] = p
                frames.append(i)

        for i, p in enumerate(prompt.prompts):
            if p is not None:
                prompts[i] = p
                frames.append(i)

        frames = list(set(frames))
        frames.sort()

        return Prompt(prompts=prompts, frames=frames)

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
        return f'Prompt(+[{self.prompt}], -[{self.negative_prompt}]) {self.frames}'

    def save(self, path):
        assert self.embeddings is not None, "Embeddings are empty"
        self.embeddings.save(path)
        return self

    def load(self, path):
        self.embeddings = PromptEmbeddings().load(path)
        return self
