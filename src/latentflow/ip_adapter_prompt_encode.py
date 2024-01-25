import torch
import logging
from einops import rearrange

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings

class IPAdapterPromptEncode(Flow):
    """
    Prompt -> IPAdapterPromptEncode -> Prompt
    """

    def __init__(self, ip_adapter):
        self.ip_adapter = ip_adapter

    def apply(self, prompt):
        logging.debug("IPAdapterPromptEncode apply %s", prompt)

        if prompt.prompts is not None:

            embeddings = []
            for i, p in enumerate(prompt.prompts):
                e = self.encode(p)
                embeddings.append(e)
            embeddings = torch.stack(embeddings)
            embeddings = rearrange(embeddings, 'f b n c -> (b f) n c')
            p.embeddings = PromptEmbeddings(embeddings)

        else:
            embeddings = self.encode(prompt)

        prompt.embeddings = PromptEmbeddings(embeddings=embeddings)
        return prompt

    def encode(self, prompt):
        cond_image_embeddings, uncond_image_embeddings = self.ip_adapter.get_image_embeds(
                images=prompt.image.chw()[0],
                negative_images=prompt.negative_image.chw()[0],
                )

        if cond_image_embeddings.shape[0] > 1:
            cond_image_embeddings = torch.cat(cond_image_embeddings.chunk(cond_image_embeddings.shape[0]), dim=1)
        if uncond_image_embeddings.shape[0] > 1:
            uncond_image_embeddings = torch.cat(uncond_image_embeddings.chunk(uncond_image_embeddings.shape[0]), dim=1)

        cond_embeddings = cond_image_embeddings
        uncond_embeddings = uncond_image_embeddings
        if prompt.embeddings is not None:
            uncond_embeddings, cond_embeddings = prompt.embeddings.embeddings.chunk(2)
            cond_embeddings = torch.cat([cond_embeddings, cond_image_embeddings], dim=1)
            uncond_embeddings = torch.cat([uncond_embeddings, uncond_image_embeddings], dim=1)

        embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        return embeddings
