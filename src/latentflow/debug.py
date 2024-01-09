import torch
import logging

from .flow import Flow

class Debug(Flow):
	def __init__(self, comment=""):
        self.comment = comment

    def apply(self, other):
        logging.debug('Debug %s %s', self.comment, other)
        return other
