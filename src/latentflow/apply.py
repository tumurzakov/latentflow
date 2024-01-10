import torch
import logging

from .flow import Flow

class Apply(Flow):
    def __init__(self, callback):
        self.callback = callback

    def apply(self, other):
        return self.callback(other)
