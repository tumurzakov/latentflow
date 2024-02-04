import torch
import logging
import importlib
import diffusers

from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.models.unet import UNet3DConditionModel

from diffusers import ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf

from .flow import Flow

class AnimateDiffPipeline(AnimationPipeline, Flow):
    @classmethod
    def load(cls,
            pretrained_model_path,
            motion_module_path,
            unet_path=None,
            text_encoder_path=None,
            vae_path=None,
            scheduler_class=None,
            controlnet_paths=[],
            motion_module_config_path=None,
            extra_tokens=None,
            *args,
            **kwargs,
            ):

        controlnet = None
        if controlnet_paths is not None:
            controlnet = []
            for controlnet_path in controlnet_paths:
                logging.debug("AnimateDiffPipelineLoad load cnet %s", controlnet_path)
                controlnet.append(
                    ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
                )

        return cls.load_pipeline(
            pretrained_model_path=pretrained_model_path,
            motion_module_path=motion_module_path,
            motion_module_config_path=motion_module_config_path,
            text_encoder_path=text_encoder_path,
            unet_path=unet_path,
            vae_path=vae_path,
            scheduler_class=scheduler_class,
            extra_tokens=extra_tokens,
            controlnet=controlnet,
            *args,
            **kwargs,
        )

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs

    def apply(self, other):
        kwargs = self.call_kwargs

        logging.debug("AnimateDiffPipeline %s %s",
                latent, type(self))

        if isinstance(latent, Latent) and 'latents' not in kwargs:
            kwargs['latents'] = latent.latent.to(
                    device=self.unet.device,
                    dtype=self.unet.dtype)

        kwargs['output_type'] = 'latent'

        result = self.__orig_call__(**kwargs)

        if isinstance(result, tuple):
            result = result[0]

        return Latent(result)

    @classmethod
    def load_pipeline(cls,
        pretrained_model_path,
        motion_module_path,
        motion_module_config_path=None,
        text_encoder_path = None,
        unet_path = None,
        vae_path = None,
        scheduler_class_name = None,
        scheduler_config_extra = None,
        extra_tokens = None,
        controlnet = [],
        *args,
        **kwargs,
    ):
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
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, subfolder="text_encoder").to('cuda')

        try:
            logging.debug("AnimateDiffPipelineLoad vae %s", vae_path)
            vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae").to('cuda')
        except:
            logging.debug("AnimateDiffPipelineLoad vae %s", vae_path)
            vae = AutoencoderKL.from_pretrained(vae_path).to('cuda')

        logging.debug("AnimateDiffPipelineLoad unet %s", unet_path)
        unet = UNet3DConditionModel.from_pretrained_2d(
                unet_path,
                subfolder="unet",
                unet_additional_kwargs=OmegaConf.to_container(motion_module_config.unet_additional_kwargs))

        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

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

        cls.__orig_call__ = cls.__call__

        pipeline = cls(
                unet=unet,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                scheduler=scheduler,
                controlnet=controlnet if controlnet is not None and len(controlnet) > 0 else None,
                )

        logging.debug("AnimateDiffPipelineLoad motion_module %s", motion_module_path)
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        pipeline.unet = pipeline.unet.half()

        return pipeline.to('cuda')