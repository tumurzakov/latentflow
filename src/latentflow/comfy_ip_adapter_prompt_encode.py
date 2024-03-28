import torch
import logging
import gc
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from .flow import Flow
from .prompt_embeddings import PromptEmbeddings
from .debug import tensor_hash

from diffusers.pipelines.controlnet import MultiControlNetModel

import comfy
from comfy.model_patcher import ModelPatcher
from comfy.model_management import cleanup_models
from .comfy_ip_adapter_plus import (
    IPAdapterModelLoader,
    IPAdapterApply,
    IPAdapterEncoder,
    InsightFaceLoader,
)

from .ip_adapter_attention_processor import MultiAttnProcessor

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from .ip_adapter_attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .ip_adapter_attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor


class ComfyIPAdapterPromptEncode(Flow):
    """
    Prompt -> IPAdapterPromptEncode -> Prompt
    """

    def __init__(self,
            ip_adapter_path=None,
            ip_adapter=None,
            clip_vision_path=None,
            clip_vision=None,
            insightface=None,
            noise=0.0,
            pipe=None,
            weight=1.0,
            onload_device='cuda',
            offload_device='cpu',
            ):

        self.ip_adapter_path = ip_adapter_path
        self.ip_adapter = ip_adapter
        self.clip_vision_path = clip_vision_path
        self.clip_vision = clip_vision
        self.insightface = insightface
        self.noise = noise
        self.pipe = pipe
        self.weight = weight
        self.onload_device = onload_device
        self.offload_device = offload_device

        self.model = None

        if self.pipe is not None:
            self.load()
            self.set_ip_adapter()

    def load(self):
        if self.ip_adapter is None and self.ip_adapter_path is not None:
            self.ip_adapter = IPAdapterModelLoader().load_ipadapter_model(self.ip_adapter_path)[0]

        if self.clip_vision is None and self.clip_vision_path is not None:
            self.clip_vision = comfy.clip_vision.load(self.clip_vision_path)

        if self.pipe is not None:
            self.device = self.pipe.unet.device
            self.dtype = self.pipe.unet.dtype
            self.model = ModelPatcher(self.pipe.unet,
                    load_device=self.onload_device,
                    offload_device=self.offload_device)

        self.is_portrait = "proj.2.weight" in self.ip_adapter["image_proj"] and \
                not "proj.3.weight" in self.ip_adapter["image_proj"] and \
                not "0.to_q_lora.down.weight" in self.ip_adapter["ip_adapter"]

        self.is_faceid = self.is_portrait or "0.to_q_lora.down.weight" in self.ip_adapter["ip_adapter"]

        if self.insightface is None and self.is_faceid and self.clip_vision is not None:
            self.insightface = InsightFaceLoader().load_insight_face(self.onload_device.upper())[0]

        if self.clip_vision is not None:
            self.ip_adapter_apply = IPAdapterApply()
            self.ip_adapter_apply.init(
                    self.ip_adapter,
                    model=self.model,
                    weight=self.weight,
                    clip_vision=self.clip_vision,
                    insightface=self.insightface,
                    )


        self.tensor_cache = {}

    def onload(self):
        pass

    def offload(self):
        pass

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
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(unet.device, dtype=unet.dtype)

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

    def __call__(self, video, weight=None, scale=None):
        self.video = video

        if weight is not None:
            self.weight = weight

        if scale is not None:
            self.set_scale(scale)

        return self

    def apply(self, prompt):
        logging.debug("ComfyIPAdapterPromptEncode apply %s", prompt)

        self.onload()
        prompt.onload()
        self.video.onload()

        if prompt.prompts is not None:
            self.tensor_cache = {}

            last = None
            embeddings = []
            for i, p in enumerate(tqdm(prompt.prompts, desc='ip adapter')):
                image = self.video.hwc()[0][i:i+1]/255.0
                h = tensor_hash(image)
                if h in self.tensor_cache:
                    e = self.tensor_cache[h]
                else:
                    try:
                        e = self.encode(p.embeddings.embeddings, image)
                    except:
                        e = last

                    self.tensor_cache[h] = e

                last = e

                if p is not None:
                    p.embeddings = PromptEmbeddings(e)

                embeddings.append(e)

            length = len(prompt.prompts)
            bs_embed, seq_len, _ = embeddings[0].shape
            embeddings = torch.cat(embeddings, dim=1)
            embeddings = embeddings.view(bs_embed * length, seq_len, -1)

            self.tensor_cache = {}
        else:
            image = self.video.hwc()[0][:1]/255.0
            embeddings = self.encode(prompt.embeddings.embeddings, image)

        prompt.embeddings = PromptEmbeddings(embeddings)

        self.video.offload()
        prompt.offload()
        self.offload()

        return prompt

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    @torch.inference_mode()
    def encode(self, embeddings, image):

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
        if embeddings is not None:
            uncond_embeddings, cond_embeddings = embeddings.chunk(2)
            cond_embeddings = torch.cat([cond_embeddings, cond_image_embeddings], dim=1)
            uncond_embeddings = torch.cat([uncond_embeddings, uncond_image_embeddings], dim=1)

        return torch.cat([uncond_embeddings, cond_embeddings])
