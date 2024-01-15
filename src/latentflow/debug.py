import torch
import hashlib
import logging

from .flow import Flow

class Debug(Flow):
    def __init__(self, comment="", callback=None):
        self.comment = comment
        self.callback = callback

    def apply(self, other):
        debug = other

        if self.callback is not None:
            debug = self.callback(debug)

        logging.debug('Debug %s %s', self.comment, debug)
        return other

class DebugHash(Flow):
    def __init__(self, callback):
        self.callback = callback

    def apply(self, other):

        tensor = other
        if self.callback:
            tensor = self.callback(other)

        assert isinstance(tensor, torch.Tensor)

        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        hash_object = hashlib.sha256(tensor_bytes)
        hash_value = hash_object.hexdigest()
        logging.debug('DebugHash %s %s', tensor.shape, hash_value)

        return other

