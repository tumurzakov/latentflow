import logging
import torch
from .flow import Flow

class PromptEmbeddings(Flow):

    def __init__(self, embeddings=None):
        # torch.Size([2, 77, 768])
        self.embeddings = embeddings

    def slice(self, l):
        logging.debug(f"PromptEmbeddings slice {l}")
        uncond_embeddings, cond_embeddings = self.embeddings.chunk(2)
        uncond = uncond_embeddings[l,:,:]
        cond = cond_embeddings[l,:,:]
        embeddings = torch.cat([uncond, cond])
        return PromptEmbeddings(embeddings)

    def save(self, path):
        torch.save(self.embeddings, path)
        return self

    def load(self, path):
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

