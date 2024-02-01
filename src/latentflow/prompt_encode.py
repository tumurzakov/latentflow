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
        logging.debug(f"CompelPromptEncode apply {prompt}")

        if prompt.prompts is not None:

            embeddings = []
            for i, p in enumerate(prompt.prompts):
                e = self.encode(
                        prompt=p.prompt,
                        negative_prompt=p.negative_prompt,
                        )
                embeddings.append(e)
                p.embeddings = PromptEmbeddings(e)

            length = len(prompt.prompts)
            bs_embed, seq_len, _ = embeddings[0].shape
            embeddings = torch.cat(embeddings, dim=1)
            embeddings = embeddings.view(bs_embed * length, seq_len, -1)

        else:
            embeddings = self.encode(
                    prompt=prompt.prompt,
                    negative_prompt=prompt.negative_prompt,
                    )

        prompt.embeddings = PromptEmbeddings(embeddings=embeddings)
        return prompt

