import torch
import logging

from .flow import Flow

class Seed(Flow):
    def __init__(self, seed=None):
        if seed is None:
            seed = torch.seed()

        self.seed = int(seed)
        torch.manual_seed(self.seed)

        logging.debug(f'{self}')

    def apply(self, other):
        return self

    def __str__(self):
        return f'Seed({self.seed})'
