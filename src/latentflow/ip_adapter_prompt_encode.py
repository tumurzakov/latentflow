import torch
import logging
from einops import rearrange

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings

class IPAdapterPromptEncode(Flow):
    """
    Prompt -> IPAdapterPromptEncode -> PromptEmbeddings
    """

    def __init__(self, ip_adapter, scale=None):
        self.ip_adapter = ip_adapter
        self.scale = scale

    def apply(self, prompt):
        logging.debug("IPAdapterPromptEncode apply %s", prompt)

        if prompt.prompts is not None:

            embeddings = []
            for i, p in enumerate(prompt.prompts):
                e = self.encode(p, self.scale)
                embeddings.append(e)
            embeddings = torch.stack(embeddings)
            embeddings = rearrange(embeddings, 'f b n c -> (b f) n c')

        else:
            embeddings = self.encode(prompt, self.scale)

        return PromptEmbeddings(embeddings=embeddings)

    def encode(self, prompt, scale=None):
        if scale is not None:
            self.ip_adapter.set_scale(scale)

        text_embeddings, uncond_embeddings = self.ip_adapter.get_prompt_embeds(
                images=prompt.image.chw()[0],
                negative_images=prompt.negative_image.chw()[0],
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                )

        embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return embeddings
