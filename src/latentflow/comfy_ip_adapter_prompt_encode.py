import torch
import logging
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings

from diffusers.pipelines.controlnet import MultiControlNetModel

import comfy
from comfy.model_patcher import ModelPatcher
from .comfy_ip_adapter_plus import (
    IPAdapterModelLoader,
    IPAdapterApply,
    IPAdapterEncoder,
    InsightFaceLoader,
)

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from .ip_adapter_attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .ip_adapter_attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor


class ComfyIPAdapterPromptEncode(Flow):
    """
    Prompt -> IPAdapterPromptEncode -> Prompt
    """

    def __init__(self,
            ip_adapter_path,
            clip_path=None,
            noise=0.0,
            pipe=None,
            weight=1.0,
            ):

        self.ip_adapter_path = ip_adapter_path
        self.image_encoder_path = clip_path
        self.noise = noise
        self.pipe = pipe
        self.weight = weight
        self.device = self.pipe.unet.device
        self.dtype = self.pipe.unet.dtype

        self.model = ModelPatcher(pipe.unet, load_device='cuda', offload_device='cpu')
        self.ip_adapter = IPAdapterModelLoader().load_ipadapter_model(ip_adapter_path)[0]
        self.clip_vision = None
        if clip_path is not None:
            self.clip_vision = comfy.clip_vision.load(clip_path)

        is_portrait = "proj.2.weight" in self.ip_adapter["image_proj"] and \
                not "proj.3.weight" in self.ip_adapter["image_proj"] and \
                not "0.to_q_lora.down.weight" in self.ip_adapter["ip_adapter"]

        is_faceid = is_portrait or "0.to_q_lora.down.weight" in self.ip_adapter["ip_adapter"]

        self.insightface = None
        if is_faceid:
            self.insightface = InsightFaceLoader().load_insight_face('CUDA')[0]

        self.ip_adapter_apply = IPAdapterApply()
        self.ip_adapter_apply.init(
                self.ip_adapter,
                model=self.model,
                weight=self.weight,
                clip_vision=self.clip_vision,
                insightface=self.insightface,
                )

        self.set_ip_adapter()

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet") and self.pipe.controlnet is not None:
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor())
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor())

        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        state_dict = {}
        for k in self.ip_adapter["ip_adapter"]:
            if "lora" not in k:
                state_dict[k] = self.ip_adapter["ip_adapter"][k]
        ip_layers.load_state_dict(state_dict)

        return self

    def apply(self, prompt):
        logging.debug("ComfyIPAdapterPromptEncode apply %s", prompt)

        if prompt.prompts is not None:

            embeddings = []
            for i, p in enumerate(tqdm(prompt.prompts, desc='ip adapter')):
                e = self.encode(p)
                p.embeddings = PromptEmbeddings(e)
                embeddings.append(e)
            embeddings = torch.stack(embeddings)
            embeddings = rearrange(embeddings, 'f b n c -> (b f) n c')

        else:
            embeddings = self.encode(prompt)

        prompt.embeddings = PromptEmbeddings(embeddings)
        return prompt

    @torch.inference_mode()
    def encode(self, prompt):

        image = prompt.image.hwc()[0]/255.0

        cond_image_embeddings, uncond_image_embeddings = self.ip_adapter_apply.encode(
                self.ip_adapter,
                model=self.model,
                weight=self.weight,
                clip_vision=self.clip_vision,
                image=image,
                noise=self.noise,
                insightface=self.insightface,
                )

        cond_embeddings = cond_image_embeddings
        uncond_embeddings = uncond_image_embeddings
        if prompt.embeddings is not None:
            uncond_embeddings, cond_embeddings = prompt.embeddings.embeddings.chunk(2)
            cond_embeddings = torch.cat([cond_embeddings, cond_image_embeddings], dim=1)
            uncond_embeddings = torch.cat([uncond_embeddings, uncond_image_embeddings], dim=1)

        return torch.cat([uncond_embeddings, cond_embeddings])
