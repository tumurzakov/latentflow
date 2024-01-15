import torch
import logging

from .flow import Flow

class Noop(Flow):
    def __init__(self):
        logging.debug("Noop init")

    def apply(self, other):
        logging.debug("Noop apply %s", type(other))

        return other
