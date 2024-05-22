import os
import logging
import torch
from .flow import Flow

class PromptEmbeddings(Flow):

    def __init__(self,
            embeddings=None,
            onload_device='cuda',
            offload_device='cpu',
            ):
        # torch.Size([2, 77, 768])
        self.embeddings = embeddings
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.embeddings = self.embeddings.to(self.onload_device)

    def offload(self):
        self.embeddings = self.embeddings.to(self.offload_device)

    def slice(self, l, do_classifier_free_guidance=True):
        logging.debug(f"PromptEmbeddings slice {l} {self}")

        if do_classifier_free_guidance:
            uncond_embeddings, cond_embeddings = self.embeddings.chunk(2)
            uncond = uncond_embeddings[l,:,:]
            cond = cond_embeddings[l,:,:]
            embeddings = torch.cat([uncond, cond])
        else:
            embeddings = self.embeddings[l,:,:]

        return PromptEmbeddings(embeddings)

    def save(self, path):
        torch.save(self.embeddings, path)
        return self

    def load(self, path):

        if os.path.isfile(path):
            self.embeddings = torch.load(path)

        return self

    def __getitem__(self, key):
        return PromptEmbeddings(self.embeddings[key])

    def __str__(self):
        if self.embeddings is not None:
            shape = self.embeddings.shape
            device = self.embeddings.device
            dtype = self.embeddings.dtype
            return f'PromptEmbeddings({shape}, {device}, {dtype})'

        return f'PromptEmbeddings(None)'

