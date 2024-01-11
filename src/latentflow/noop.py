import torch
import logging

from .flow import Flow

class Noop(Flow):
    def apply(self, other):
        return self
