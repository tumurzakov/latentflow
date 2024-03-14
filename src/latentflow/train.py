import torch
import logging

from .flow import Flow

class TrainPredict(Flow):
    def __init__(self, *args, **kwargs): ...

class TrainCalcLoss(Flow):
    def __init__(self, *args, **kwargs): ...

class TrainStep(Flow):
    def __init__(self, *args, **kwargs): ...
