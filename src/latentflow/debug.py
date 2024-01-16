import torch
import hashlib
import logging

from .flow import Flow

def tensor_hash(tensor):
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    hash_object = hashlib.sha256(tensor_bytes)
    hash_value = hash_object.hexdigest()
    return hash_value

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
    def __init__(self, comment="", callback=None):
        self.callback = callback
        self.comment = comment

    def apply(self, other):

        tensor = other
        if self.callback is not None:
            tensor = self.callback(other)

        assert isinstance(tensor, torch.Tensor)

        hash_value = tensor_hash(tensor)

        logging.debug('DebugHash %s %s %s',
                self.comment, tensor.shape, hash_value)

        return other

