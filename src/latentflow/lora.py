import torch
import logging
import hashlib

from .flow import Flow

class LoraOn(Flow):
    def __init__(self, loras: dict = {}, pipe=None, fuse=False):
        self.loras = loras
        self.pipe = pipe
        self.fuse = fuse

        self.lora_adapters = []
        self.lora_weights = []

        logging.debug('LoraOn init %s, fuse=%s', loras, fuse)


    def onload(self):
        self.pipe.enable_lora()
        self.pipe.unload_lora_weights()

        for lora in self.loras:
            scale = self.loras[lora]
            name = hashlib.sha1(lora.encode()).hexdigest()
            self.pipe.load_lora_weights(lora, adapter_name=name)
            self.lora_adapters.append(name)
            self.lora_weights.append(scale)
            logging.debug('LoraOn load %s %f', lora, scale)

        if self.fuse:
            self.pipe.fuse_lora()

    def offload(self):
        self.pipe.enable_lora()
        self.pipe.unload_lora_weights()

    def apply(self, other=None):
        logging.debug("LoraOn apply %s %s", self.loras, type(other))

        self.onload()

        self.pipe.set_adapters(self.lora_adapters, self.lora_weights)

        logging.debug("LoraOn active %s", self.pipe.get_active_adapters())

        self.offload()

        return other

class LoraOff(Flow):
    def __init__(self, pipe=None):
        self.pipe = pipe
        logging.debug('LoraOff init')

    def apply(self, other):
        logging.debug('LoraOff apply %s', type(other))

        self.pipe.unload_lora_weights()

        return other
