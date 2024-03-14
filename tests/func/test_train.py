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

class TestTrain(unittest.TestCase):
    def test_train_lora(self):

        models = os.getenv('MODELS_DIR')
        pretrained_model_path = os.getenv('PRETRAINED_MODEL_NAME_OR_PATH')
        samples_dir = 'train_samples'

        state = lf.State({})
        state['pipe'] = diffusers.StableDiffusionPipeline.from_pretrained(pretrained_model_path)

        (lf.Unet(
                unet=state['pipe'].unet,
                scheduler=state['pipe'].scheduler,
                fp16=True,
            )
            | lf.LoraInitTrain(rank=4)
            > state('unet')
            )

        (lf.Dataset(samples_dir=samples_dir)
            | lf.TrainPredict(state['unet'])
            | lf.TrainCalcLoss()
            | lf.TrainStep(state['unet'])
        )



