import torch
import logging

from .flow import Flow

class Tensor(Flow):
    def __init__(self,
            tensor,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):
        self.tensor = tensor
        self.shape = tensor.shape
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.tensor = self.tensor.to(self.onload_device)

    def offload(self):
        self.tensor = self.tensor.to(self.offload_device)

    def apply(self, other):
        return self

    def save(self, path):
        torch.save(self.tensor, path)
        return self

    def load(self, path, device='cpu'):
        return Tensor(torch.load(path).to(device))

    def __getitem__(self, key):
        return Tensor(self.tensor[key])

    def __str__(self):
        return f'Tensor({self.tensor.shape}, {self.tensor.device}, {self.tensor.dtype})'

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
