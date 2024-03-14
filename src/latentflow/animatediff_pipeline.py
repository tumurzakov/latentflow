import torch
import logging
import importlib
import diffusers

from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.models.unet import UNet3DConditionModel

from diffusers import ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf

from diffusers.models.attention_processor import AttnProcessor2_0
import hashlib
#from safetensors import safe_open

from .flow import Flow
from .latent import Latent
from .lora import LoraOn

class AnimateDiffPipeline(AnimationPipeline, Flow):
    onload_device = 'cuda'
    offload_device = 'cpu'

    @classmethod
    def load(cls,
            pretrained_model_path,
            motion_module_path,
            #dreambooth_path=None,
            unet_path=None,
            text_encoder_path=None,
            vae_path=None,
            vae_is_svd = False,
            scheduler_class=None,
            controlnet_paths=[],
            motion_module_config_path=None,
            extra_tokens=None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            loras={},
            *args,
            **kwargs,
            ):

        dtype = torch.float32
        if 'fp16' in kwargs:
            dtype = torch.float16

        controlnet = None
        if controlnet_paths is not None:
            controlnet = []
            for controlnet_path in controlnet_paths:
                logging.debug("AnimateDiffPipelineLoad load cnet %s", controlnet_path)
                controlnet.append(
                    ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
                )

        return cls.load_pipeline(
            pretrained_model_path=pretrained_model_path,
            motion_module_path=motion_module_path,
            motion_module_config_path=motion_module_config_path,
            #dreambooth_path=dreambooth_path,
            text_encoder_path=text_encoder_path,
            unet_path=unet_path,
            vae_path=vae_path,
            vae_is_svd=vae_is_svd,
            scheduler_class=scheduler_class,
            extra_tokens=extra_tokens,
            controlnet=controlnet,
            loras=loras,
            onload_device=onload_device,
            offload_device=offload_device,
            *args,
            **kwargs,
        )

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return self

    def onload(self):
        self.unet = self.unet.to(self.onload_device)
        self.text_encoder = self.text_encoder.to(self.onload_device)
        self.vae = self.vae.to(self.onload_device)

    def offload(self):
        self.unet = self.unet.to(self.offload_device)
        self.text_encoder = self.text_encoder.to(self.offload_device)
        self.vae = self.vae.to(self.offload_device)

    def apply(self, latent: Latent) -> Latent:
        kwargs = self.call_kwargs

        self.onload()
        latent.onload()

        logging.debug("AnimateDiffPipeline %s %s",
                latent, type(self))

        if isinstance(latent, Latent) \
                and latent.latent is not None \
                and 'latents' not in kwargs:
            kwargs['latents'] = latent.latent.to(
                    device=self.unet.device,
                    dtype=self.unet.dtype)

        kwargs['output_type'] = 'latent'

        with torch.no_grad():
            result = self.__orig_call__(**kwargs)

        if isinstance(result, tuple):
            result = result[0]

        logging.debug("Pipeline result %s", result.shape)

        result.offload()
        latent.offload()
        self.offload()

        return Latent(result)

    @classmethod
    def load_pipeline(cls,
        pretrained_model_path,
        motion_module_path,
        motion_module_config_path=None,
        #dreambooth_path=None,
        text_encoder_path = None,
        unet_path = None,
        vae_path = None,
        vae_is_svd = False,
        scheduler_class_name = None,
        scheduler_config_extra = None,
        extra_tokens = None,
        controlnet = [],
        loras = {},
        onload_device: str='cuda',
        offload_device: str='cpu',
        *args,
        **kwargs,
    ):
        dtype = torch.float32
        if 'fp16' in kwargs:
            dtype = torch.float16

        if motion_module_config_path is None:
            motion_module_config_path = f'{motion_module_path}/inference.yaml'

        motion_module_config = OmegaConf.load(motion_module_config_path)
        if 'motion_module_config' in kwargs:
            kwargs_config = OmegaConf.create(kwargs['motion_module_config'])
            motion_module_config = OmegaConf.merge(motion_module_config, kwargs_config)

        if text_encoder_path is None:
          text_encoder_path = pretrained_model_path

        if unet_path is None:
            unet_path = pretrained_model_path

        if vae_path is None:
            vae_path = pretrained_model_path

        logging.debug("AnimateDiffPipelineLoad load tokenizer %s", pretrained_model_path)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
        if extra_tokens is not None:
            added = tokenizer.add_tokens(extra_tokens)
            logging.info("Extra tokens added: %s %d", extra_tokens, added)

        logging.debug("AnimateDiffPipelineLoad text_encoder %s", text_encoder_path)
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, subfolder="text_encoder")

        if vae_is_svd:
            logging.debug("AnimateDiffPipelineLoad vae %s", vae_path)
            vae = AutoencoderKLTemporalDecoder.from_pretrained(vae_path, subfolder="vae")
        else:
            try:
                logging.debug("AnimateDiffPipelineLoad vae %s", vae_path)
                vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
            except:
                logging.debug("AnimateDiffPipelineLoad vae %s", vae_path)
                vae = AutoencoderKL.from_pretrained(vae_path)

        logging.debug("AnimateDiffPipelineLoad unet %s", unet_path)

        unet = UNet3DConditionModel.from_pretrained_2d(
                unet_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(motion_module_config.unet_additional_kwargs))

        unet.set_attn_processor(AttnProcessor2_0())

        #if is_xformers_available() and ('xformers' not in kwargs or kwargs['xformers']):
        #    unet.enable_xformers_memory_efficient_attention()

        if scheduler_class_name is None:
            scheduler_class_name = 'DDIMScheduler'

        scheduler_config = {
            'num_train_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': "linear",
            'steps_offset': 1,
        }

        if scheduler_config_extra is not None and isinstance(scheduler_config_extra, dict):
            scheduler_config.update(scheduler_config_extra)

        if scheduler_class_name == 'DDIMScheduler':
            scheduler_config['clip_sample'] = False

        scheduler_class = getattr(diffusers, scheduler_class_name)
        scheduler = scheduler_class(**scheduler_config)

        cls.__orig_call__ = AnimationPipeline.__call__

        pipeline = cls(
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                scheduler=scheduler,
                controlnet=controlnet if controlnet is not None and len(controlnet) > 0 else None,
                )

        #if dreambooth_path is not None:
        #    pipeline = cls.load_dreambooth(pipeline, dreambooth_path)

        pipeline.onload_device = onload_device
        pipeline.offload_device = offload_device

        if len(loras) > 0:
            LoraOn(loras, pipeline, fuse=True).apply()

        logging.debug("AnimateDiffPipelineLoad motion_module %s", motion_module_path)
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        if 'fp16' in kwargs and kwargs['fp16']:
            pipeline.unet = pipeline.unet.half()

        if 'compile_unet' in kwargs and kwargs['compile_unet']:
            unet = torch.compile(unet, mode="reduce-overhead", fullgraph=True)

        return pipeline

    #@classmethod
    #def load_dreambooth(cls, pipeline, dreambooth_path):

    #    assert dreambooth_path.endswith(".safetensors"), 'dreambooth_path must end with safetensors'

    #    state_dict = {}
    #    with safe_open(dreambooth_path, framework="pt", device="cpu") as f:
    #        for key in f.keys():
    #            state_dict[key] = f.get_tensor(key)

    #    base_state_dict = state_dict

    #    # vae
    #    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
    #    pipeline.vae.load_state_dict(converted_vae_checkpoint)

    #    # unet
    #    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
    #    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)

    #    # text_model
    #    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

    #    return pipeline

