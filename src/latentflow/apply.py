import torch
import logging

from .flow import Flow

class Apply(Flow):
    def __init__(self, callback):
        self.callback = callback

    def apply(self, other):
        return self.callback(other)

class Call(Flow):
    def __init__(self, callback, *args):
        self.callback = callback
        self.args = args

    def apply(self, other):
        self.callback(*self.args)
        return other
