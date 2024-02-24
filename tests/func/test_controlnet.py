import os
import sys

import unittest
import torch
import diffusers
import gc
from einops import rearrange

import logging
logging.basicConfig(level=logging.DEBUG)

import latentflow as lf

class TestControlnet(unittest.TestCase):
    def test_apply(self):
        state = lf.State({
            'video_length': 16,
            'height': 96,
            'width': 96,
            'num_inference_steps': 10,
            'guidance_scale': 7.0,
            })

        models = os.getenv('MODELS_DIR')
        pretrained_model_path = os.getenv('PRETRAINED_MODEL_NAME_OR_PATH')

        (state
            | lf.DebugCUDAUsage("MEMORY pipe init")
            )

        (lf.AnimateDiffPipeline.load(
                pretrained_model_path=pretrained_model_path,
                motion_module_path=f'{models}/mm_sd_v15_v3.ckpt',
                motion_module_config_path=f'{models}/mm_sd_v15_v3.yaml',
                controlnet_paths=[
                    'lllyasviel/control_v11p_sd15_lineart',
                    ],
                scheduler_class_name='EulerDiscreteScheduler',
            )
            | lf.DebugCUDAUsage("MEMORY pipe init")
            > state('pipe')
            )

        (lf.Unet(
                unet=state['pipe'].unet,
                scheduler=state['pipe'].scheduler,
                fp16=True,
            )
            | lf.DebugCUDAUsage("MEMORY unet init")
            > state('unet')
            )

        (lf.ControlNet(
            controlnet=state['pipe'].controlnet,
            scheduler=state['pipe'].scheduler,
            )
            | lf.DebugCUDAUsage("MEMORY cnet init")
            > state('cnet')
            )

        (state
            | lf.Schedule(
                scheduler=state['pipe'].scheduler,
                num_inference_steps=state['num_inference_steps'],
                strength=state['strength'],
                )
            > state('timesteps')
            )

        video = lf.Video('HWC', torch.zeros((1,state['video_length'],state['height'],state['width'],3)))

        (lf.Prompt(
                prompt="prompt",
                negative_prompt="negative prompt",
                frames=list(range(0,state['video_length'])),
            )
            | lf.CompelPromptEncode(tokenizer=state['pipe'].tokenizer, text_encoder=state['pipe'].text_encoder)
            | lf.DebugCUDAUsage("MEMORY prompt")
            > state('prompt')
            )

        latent = lf.Latent(torch.zeros((1,4,state['video_length'],state['height']//8,state['width']//8)))
        (latent
            | lf.CFGPrepare(guidance_scale=state['guidance_scale'])
            | lf.DebugCUDAUsage("MEMORY before cnet")
            | state['cnet'](
                timestep_index=0,
                timestep=state['timesteps'].timesteps[0],
                image=[
                    video.cnet().tensor,
                    ],
                timesteps=state['timesteps'],
                embeddings=state['prompt'].embeddings,
            )
            | lf.DebugCUDAUsage("MEMORY before unet")
            | state['unet'](
                timestep=state['timesteps'].timesteps[0],
                embeddings=state['prompt'].embeddings,
            )
            | lf.DebugCUDAUsage("MEMORY after unet")
            )

