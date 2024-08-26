import torch
import logging
from .flow import Flow
from .meta_utils import read_meta, write_meta

class Meta(Flow):
    def __init__(self, meta):
        self.meta = meta

    def apply(self, other):
        if isinstance(other, dict):
            self.meta.update(other)

        elif isinstance(other, Meta):
            self.meta.update(other.meta)

        return self

    def save(self, path):
        logging.debug(f"Meta save {path}")
        write_meta(self.meta, path)

    def load(self, path):
        self.meta = read_meta(path)

