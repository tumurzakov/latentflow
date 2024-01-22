import logging
from .flow import Flow

class PromptEmbeddings(Flow):

    def __init__(self, embeddings=None):
        # torch.Size([2, 77, 768])
        self.embeddings = embeddings
        logging.debug(f"PromptEmbeddings init {self}")

    def slice(self, l):
        logging.debug(f"PromptEmbeddings slice {l}")
        return PromptEmbeddings(self.embeddings[l, :, :])

    def __str__(self):
        if self.embeddings is not None:
            shape = self.embeddings.shape
            device = self.embeddings.device
            dtype = self.embeddings.dtype
            return f'PromptEmbeddings({shape}, {device}, {dtype})'

        return f'PromptEmbeddings(None)'

