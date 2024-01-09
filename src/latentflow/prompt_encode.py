import torch
import logging
from .flow import Flow
from .prompt import Prompt
from .text_embeddings import text_embeddings
from .prompt_embeddings import PromptEmbeddings

class PromptEncode(Flow):
    def __init__(self, tokenizer, text_encoder):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        logging.debug("PromptEncode init")

    @torch.no_grad()
    def apply(self, prompt: Prompt):
        logging.debug(f"PromptEncode apply {prompt}")

        cond_embeddings, uncond_embeddings = text_embeddings(
                self.tokenizer,
                self.text_encoder,
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                )

        embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        return PromptEmbeddings(embeddings=embeddings)
