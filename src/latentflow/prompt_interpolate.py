import torch
import logging
import torch.nn.functional as F

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings

def interpolate_linear(tensor1, tensor2, n_steps):
    # Ensure tensors have the same shape
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"

    # Calculate step size for interpolation
    step_size = 1.0 / (n_steps + 1)

    # Initialize list to store interpolated tensors
    interpolated_tensors = []

    # Perform interpolation
    for i in range(1, n_steps + 1):
        interpolation_weight = step_size * i
        interpolated_tensor = (1 - interpolation_weight) * tensor1 + interpolation_weight * tensor2
        interpolated_tensors.append(interpolated_tensor)

    return interpolated_tensors

class PromptInterpolate(Flow):
    def apply(self, prompt):
        prompts = prompt.prompts
        start = None

        embeddings = []
        for i, p in enumerate(prompts):
            if p is not None:
                if start is None:
                    start = i
                    embeddings.append(p.embeddings.embeddings)
                else:
                    count = i - start
                    interpolated = interpolate_linear(prompts[start].embeddings.embeddings, prompts[i].embeddings.embeddings, count)
                    embeddings = embeddings + interpolated

                    start = i

        length = len(prompt.prompts)
        bs_embed, seq_len, _ = embeddings[0].shape
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = embeddings.view(bs_embed * length, seq_len, -1)
        prompt.embeddings = PromptEmbeddings(embeddings)

        return prompt
