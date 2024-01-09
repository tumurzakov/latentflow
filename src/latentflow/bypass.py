import logging

from .flow import Flow

class Bypass(Flow):
    def __init__(self, *args, **kwargs):
        pass

    def __sub__(self, other):
        logging.debug("Bypass %s", type(other))
        return self

    def apply(self, other):
        logging.debug("Bypass apply %s", type(other))
        return other
