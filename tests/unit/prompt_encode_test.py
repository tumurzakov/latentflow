import torch

from latentflow.prompt import Prompt
from latentflow.prompt_embeddings import PromptEmbeddings

class TestPromptEncode:
    def apply(self, prompt: Prompt) -> PromptEmbeddings:
        return PromptEmbeddings(embeddings=torch.randn((1,1,1)))
