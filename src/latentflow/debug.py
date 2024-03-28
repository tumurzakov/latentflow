import torch
import hashlib
import logging

from .flow import Flow

def tensor_hash(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]

    assert isinstance(tensors, list), 'tensors must be list'

    hash_obj = hashlib.sha256()
    for tensor in tensors:
        flat_tensor = tensor.flatten()
        tensor_bytes = flat_tensor.detach().cpu().numpy().tobytes()
        hash_obj.update(tensor_bytes)
    return hash_obj.hexdigest()

class Log(Flow):
    def __init__(self, level=logging.DEBUG, comment="", callback=None):
        self.comment = comment
        self.callback = callback
        self.level = level

    def apply(self, other):
        debug = other

        if self.callback is not None:
            debug = self.callback(debug)

        logging.log(self.level,
                'Debug %s %s %.2fGB',
                self.comment,
                debug,
                torch.cuda.memory_allocated(0)/1024/1024/1024,
                )

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


class DebugCUDAUsage(Flow):
    def __init__(self, comment=""):
        self.comment = comment

    def apply(self, other):
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)

        logging.debug(f'CUDA Usage {self.comment}: allocated={allocated} reserved={reserved}')
        return other
