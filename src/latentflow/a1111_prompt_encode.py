import torch
import logging
from einops import rearrange

from transformers import CLIPTextModel, CLIPTokenizer

from .flow import Flow
from .prompt import Prompt
from .a1111_text_embeddings import text_embeddings
from .prompt_embeddings import PromptEmbeddings

class A1111PromptEncode(Flow):
    r"""
    Prompt encoder in A1111 style

    (Prompt("a1111 (prompt:1.1) style")
        | A1111PromptEncode(tokenizer, text_encoder)
        > state("prompt")
        ) >> \

    """

    def __init__(self,
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            onload_device: str='cuda',
            offload_device: str='cpu',
            clip_skip: int=1,
            ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.clip_skip = clip_skip

        logging.debug("A1111PromptEncode init")

    def onload(self):
        self.text_encoder = self.text_encoder.to(self.onload_device)

    def offload(self):
        self.text_encoder = self.text_encoder.to(self.offload_device)

    @torch.no_grad()
    def apply(self, prompt: Prompt) -> Prompt:
        logging.debug(f"CompelPromptEncode apply {prompt}")

        self.onload()
        prompt.onload()

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

        prompt.offload()
        self.offload()

        return prompt


    def encode(self, prompt, negative_prompt):
        cond_embeddings, uncond_embeddings = text_embeddings(
                self.tokenizer,
                self.text_encoder,
                prompt=prompt,
                negative_prompt=negative_prompt,
                clip_stop_at_last_layers=self.clip_skip,
                )
        embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        return embeddings



