import torch
import logging
from compel import Compel, ReturnedEmbeddingsType
from einops import rearrange
import gc
import json
from tqdm import tqdm

from .flow import Flow
from .prompt import Prompt
from .prompt_embeddings import PromptEmbeddings

class CompelPromptEncode(Flow):
    def __init__(self,
            tokenizer,
            text_encoder,
            do_classifier_free_guidance=True,
            weight=1.0,
            onload_device='cuda',
            offload_device='cpu',
            clip_skip=False,
            ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.weight = weight
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.clip_skip = clip_skip

        logging.debug("CompelPromptEncode init")

    def onload(self):
        self.text_encoder = self.text_encoder.to(self.onload_device)

        self.compel = Compel(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=self.onload_device,
            returned_embeddings_type=\
                ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED if self.clip_skip \
                    else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED,
        )

    def offload(self):
        self.text_encoder = self.text_encoder.to(self.offload_device)
        del self.compel
        gc.collect()
        torch.cuda.empty_cache()
        self.compel = None

    @torch.no_grad()
    def apply(self, prompt: Prompt):
        logging.debug(f"CompelPromptEncode apply {prompt}")

        self.onload()
        prompt.onload()

        if prompt.prompts is not None:

            self.cache = {}
            last = None

            embeddings = []
            for i, p in enumerate(tqdm(prompt.prompts)):
                if p is not None:
                    e = self.encode(
                            prompt=p.prompt,
                            negative_prompt=p.negative_prompt,
                            )
                    p.embeddings = PromptEmbeddings(e)
                    last = p
                else:
                    assert last is not None, "Last embedding must exist"
                    e = last.embeddings.embeddings

                embeddings.append(e)

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

    def encode(self, prompt, negative_prompt, num_videos_per_prompt=1):

        cache_key = json.dumps([prompt, negative_prompt])
        if cache_key in self.cache:
            return self.cache[cache_key] * self.weight

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

        self.cache[cache_key] = text_embeddings

        return text_embeddings*self.weight

