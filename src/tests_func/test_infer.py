import os
import sys

import unittest
import torch
import diffusers

import logging
logging.basicConfig(level=logging.DEBUG)

from animatediff.models.unet import UNet3DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from latentflow.state import State
from latentflow.video import Video
from latentflow.latent import Latent
from latentflow.tile import Tile
from latentflow.prompt import Prompt
from latentflow.vae_video_encode import VaeVideoEncode
from latentflow.vae_latent_decode import VaeLatentDecode
from latentflow.diffuse import Diffuse
from latentflow.unet import Unet
from latentflow.prompt_embeddings import PromptEmbeddings
from latentflow.prompt_encode import PromptEncode


class TestInfer(unittest.TestCase):
    def _test_encode_video(self):
        """
        video|VaeVideoEncode() > state("video")
        """
        pretrained_model_name_or_path = os.environ['PRETRAINED_MODEL_NAME_OR_PATH']
        _vae = diffusers.AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder='vae')

        state = State()
        video = Video('HWC', torch.randn((16,288,512,3)), device='cuda')
        latent = video|VaeVideoEncode(vae=_vae.to('cuda')) > state("latent")
        self.assertTrue(isinstance(latent, Latent))
        self.assertTrue(isinstance(state.get('latent'), Latent))

    def _test_should_encode_prompt(self):
        """
        prompt|PromptEncode() > state("embeddings")
        """
        pretrained_model_name_or_path = os.environ['PRETRAINED_MODEL_NAME_OR_PATH']
        _tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')
        _text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')

        state = State()
        embeddings = Prompt("test") | PromptEncode(tokenizer=_tokenizer, text_encoder=_text_encoder) > state("embeddings")
        self.assertTrue(isinstance(embeddings, PromptEmbeddings))
        self.assertTrue(isinstance(state.get('embeddings'), PromptEmbeddings))

    def _test_should_diffuse(self):
        """
        video|VaeVideoEncode() > state("video") >>
        prompt|PromptEncode() > state("embeddings") >>
        Diffuse(10)
        """

        unet_additional_kwargs = {
            'unet_use_cross_frame_attention': False,
            'unet_use_temporal_attention': False,
            'use_motion_module': True,
            'motion_module_resolutions': [1,2,4,8],
            'motion_module_mid_block': False,
            'motion_module_decoder_only': False,
            'motion_module_type': 'Vanilla',
            'motion_module_kwargs': {
                'num_attention_heads': 8,
                'num_transformer_block': 1,
                'attention_block_types': [
                    'Temporal_Self',
                    'Temporal_Self',
                ],
                'temporal_position_encoding': True,
                'temporal_position_encoding_max_len': 32,
                'temporal_attention_dim_div': 1,
            }
        }

        pretrained_model_name_or_path = os.environ['PRETRAINED_MODEL_NAME_OR_PATH']
        _vae = diffusers.AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder='vae')
        _scheduler = diffusers.DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler')
        _tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')
        _text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')

        state = State()
        latent = \
            (Video('HWC', torch.randn((16,288,512,3))) \
                | VaeVideoEncode(vae=_vae.to('cuda')) > state("latent")) >> \
            (Prompt("test") \
                | PromptEncode(tokenizer=_tokenizer, text_encoder=_text_encoder) > state("embeddings")) >> \
            (state | Diffuse(
                    scheduler=_scheduler,
                    num_inference_steps=10,
                    callback=lambda **kwargs: kwargs['state'].get('latent'),
                    ))

        self.assertTrue(isinstance(latent, Latent))


    def _test_should_infer(self):
        unet_additional_kwargs = {
            'unet_use_cross_frame_attention': False,
            'unet_use_temporal_attention': False,
            'use_motion_module': True,
            'motion_module_resolutions': [1,2,4,8],
            'motion_module_mid_block': False,
            'motion_module_decoder_only': False,
            'motion_module_type': 'Vanilla',
            'motion_module_kwargs': {
                'num_attention_heads': 8,
                'num_transformer_block': 1,
                'attention_block_types': [
                    'Temporal_Self',
                    'Temporal_Self',
                ],
                'temporal_position_encoding': True,
                'temporal_position_encoding_max_len': 32,
                'temporal_attention_dim_div': 1,
            }
        }

        pretrained_model_name_or_path = os.environ['PRETRAINED_MODEL_NAME_OR_PATH']
        _vae = diffusers.AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder='vae')
        _scheduler = diffusers.DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder='scheduler')
        _tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder='tokenizer')
        _text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')
        _unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_name_or_path,
                subfolder='unet',
                unet_additional_kwargs=unet_additional_kwargs,
                )

        motion_module_path = os.environ['MOTION_MODULE_PATH']
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        missing, unexpected = _unet.load_state_dict(motion_module_state_dict, strict=False)
        assert len(unexpected) == 0

        unet = Unet(unet=_unet.to('cuda'), scheduler=_scheduler)

        state = State()
        video = \
            (Video('HWC', torch.randn((16,288,512,3))) \
                | VaeVideoEncode(vae=_vae.to('cuda')) > state("latent")) >> \
            \
            (Prompt("test") \
                | PromptEncode(tokenizer=_tokenizer, text_encoder=_text_encoder.to('cuda')) > state("embeddings")) >> \
            \
            (
                    state
                    | Diffuse(
                        scheduler=_scheduler,
                        num_inference_steps=10,
                        callback=lambda timestep, state: state | unet,
                        )
                    | VaeLatentDecode(vae=_vae.to('cuda'))
            )

        self.assertTrue(isinstance(video, Video))

