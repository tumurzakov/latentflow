import torch
import logging

from .flow import Flow

class Tensor(Flow):
    def __init__(self, tensor):
        self.tensor = tensor

    def apply(self, other):
        return self

class TensorAdd(Flow):
    def __init__(self, tensor: torch.Tensor):
        self.add = tensor

    def apply(self, base: Tensor):

        base.tensor[
            0:self.add.shape[0],
            0:self.add.shape[1],
            0:self.add.shape[2],
            0:self.add.shape[3],
            0:self.add.shape[4],
            ] += self.add

        return base
