import torch
import logging
from compel import Compel
from einops import rearrange

from .flow import Flow
from .prompt import Prompt
from .prompt_embeddings import PromptEmbeddings

class CompelPromptEncode(Flow):
    def __init__(self, tokenizer, text_encoder, do_classifier_free_guidance=True):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.compel = Compel(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        logging.debug("CompelPromptEncode init")

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
            embeddings = torch.stack(embeddings)
            embeddings = rearrange(embeddings, 'f b n c -> (b f) n c')

        else:
            embeddings = self.encode(
                    prompt=prompt.prompt,
                    negative_prompt=prompt.negative_prompt,
                    )

        return PromptEmbeddings(embeddings=embeddings)

    def encode(self, prompt, negative_prompt, num_videos_per_prompt=1):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_embeddings = self.compel(prompt)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance:
            uncond_embeddings = self.compel(negative_prompt)

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

