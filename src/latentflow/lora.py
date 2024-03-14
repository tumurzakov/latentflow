import os
import torch
import logging
import hashlib
from peft import LoraConfig
from safetensors.torch import save_file, safe_open

from .flow import Flow
from .unet import Unet

class LoraOn(Flow):
    def __init__(self, loras: dict = {}, pipe=None, fuse=False):
        self.loras = loras
        self.pipe = pipe
        self.fuse = fuse

        self.lora_adapters = []
        self.lora_weights = []

        logging.debug('LoraOn init %s, fuse=%s', loras, fuse)


    def onload(self):
        self.pipe.unload_lora_weights()

        for lora in self.loras:
            scale = self.loras[lora]
            name = hashlib.sha1(lora.encode()).hexdigest()
            self.pipe.load_lora_weights(lora, adapter_name=name)
            self.lora_adapters.append(name)
            self.lora_weights.append(scale)
            logging.debug('LoraOn load %s %f', lora, scale)

    def offload(self):
        if self.fuse:
            self.pipe.unload_lora_weights()

    def apply(self, other=None):
        logging.debug("LoraOn apply %s %s", self.loras, type(other))

        self.onload()

        self.pipe.set_adapters(self.lora_adapters, self.lora_weights)

        if self.fuse:
            self.pipe.fuse_lora(adapter_names=self.lora_adapters)

        logging.debug("LoraOn active %s", self.pipe.get_active_adapters())

        self.offload()

        return other

class LoraOff(Flow):
    def __init__(self, pipe=None, unfuse=False):
        self.pipe = pipe
        self.unfuse = unfuse
        logging.debug('LoraOff init')

    def apply(self, other):
        logging.debug('LoraOff apply %s', type(other))

        if self.unfuse:
            self.pipe.unfuse_lora()

        self.pipe.unload_lora_weights()

        return other

class LoraInitTrain(Flow):
    def __init__(self, rank, unet=None):
        self.unet = unet
        self.rank = rank

    def apply(self, other):
        if isinstance(other, Unet):
            self.unet = other.unet

        unet_lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.unet.add_adapter(unet_lora_config)

        return other


class LoraMerge(Flow):
    def __init__(self, loras, file):
        self.loras = loras
        self.file = file

    def apply(self, other):
        if not os.path.isfile(self.file):
            self.merge()

        return other

    def merge(self):
        state_dict = {}
        for lora in self.loras:
            scale = self.loras[lora]
            with safe_open(lora, framework="pt", device='cpu') as f:
                for key in f.keys():
                    if key not in state_dict:
                        state_dict[key] = f.get_tensor(key) * scale
                    else:
                        state_dict[key] += f.get_tensor(key) * scale


        save_file(state_dict, self.file)
