import torch
import logging

from .flow import Flow

class ResAdapter(Flow):
    def __init__(self, model_path, lora_path, pipe):
        self.model_path = model_path
        self.lora_path = lora_path
        self.pipe = pipe
        self.load()

    def load(self):
        # Load resolution normalization
        norm_state_dict = {}
        with safe_open(self.model_path) as f:
            for key in f.keys():
                norm_state_dict[key] = f.get_tensor(key)
        m, u = self.pipe.unet.load_state_dict(norm_state_dict, strict=False)

        # Load resolution lora
        self.pipe.load_lora_weights(self.lora_path, adapter_name="res_adapter")
        self.pipe.fuse_lora(adapter_names=['res_adapter'])

        return pipeline


