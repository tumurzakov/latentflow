import torch
import hashlib
import logging

from .flow import Flow

def tensor_hash(tensor):
    tensor_bytes = tensor.detach().cpu().numpy().tobytes()
    hash_object = hashlib.sha256(tensor_bytes)
    hash_value = hash_object.hexdigest()
    return hash_value

class Log(Flow):
    def __init__(self, level=logging.DEBUG, comment="", callback=None):
        self.comment = comment
        self.callback = callback
        self.level = level

    def apply(self, other):
        debug = other

        if self.callback is not None:
            debug = self.callback(debug)

        logging.log(self.level, 'Debug %s %s', self.comment, debug)
        return other

class Debug(Log):
    def __init__(self, comment="", callback=None):
        super().__init__(
                level=logging.DEBUG,
                comment=comment,
                callback=callback,
                )

class Info(Log):
    def __init__(self, comment="", callback=None):
        super().__init__(
                level=logging.INFO,
                comment=comment,
                callback=callback,
                )

class Error(Log):
    def __init__(self, comment="", callback=None):
        super().__init__(
                level=logging.ERROR,
                comment=comment,
                callback=callback,
                )


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

