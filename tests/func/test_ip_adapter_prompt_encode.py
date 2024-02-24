import os
import sys

import unittest
import torch
import diffusers
import gc

import logging
logging.basicConfig(level=logging.DEBUG)

import latentflow as lf

class TestIpAdapterPromptEncode(unittest.TestCase):
    def test_apply(self):
        state = lf.State({
            'video_length': 16,
            })

        models = os.getenv('MODELS_DIR')
        pretrained_model_path = os.getenv('PRETRAINED_MODEL_NAME_OR_PATH')
        state['pipe'] = diffusers.StableDiffusionPipeline.from_pretrained(pretrained_model_path)

        lf.ComfyIPAdapterPromptEncode(
            ip_adapter_path=f'{models}/IP-Adapter/models/ip-adapter-faceid-plusv2_sd15.bin',
            clip_vision_path=f'{models}/IP-Adapter/models/image_encoder/pytorch_model.bin',
            pipe = state['pipe'],
            noise = 0.0,
        ) | lf.Set(state, 'comfy_ip_adapter_faceid')

        (lf.LoadImage([
                'assets/face.jpg',
            ])
            | lf.Apply(lambda x: lf.Video('HWC', x.image.unsqueeze(0)))
            | lf.VideoFaceCrop(padding_percent=0.4, size=(224,224))
            | lf.Apply(lambda x: lf.Video('HWC', torch.cat([
                v.repeat(48,1,1,1).unsqueeze(0) for v in x.video[0]
                ])))
            > state('face_video_portrait')
        )

        (lf.Prompt(
                prompt="prompt",
                negative_prompt="negative prompt",
                frames=list(range(0,state['video_length'])),
            )
            | lf.CompelPromptEncode(tokenizer=state['pipe'].tokenizer, text_encoder=state['pipe'].text_encoder)
            | state['comfy_ip_adapter_faceid'](state['face_video_portrait'][0:1], 1.0)
            | lf.DebugCUDAUsage("MEMORY")
            )
