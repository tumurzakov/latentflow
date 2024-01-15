import logging
import torch
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow

class Loop(Flow):

    def __init__(self,
            collection,
            callback:Optional[Callable] = None,
            name: Optional[str] = "",
            ):

        self.collection = collection
        if isinstance(self.collection, list):
            self.collection = iter(self.collection)

        self.callback = callback
        self.name = name

        logging.debug("Loop init %s %s", self.name, type(self.collection))

    def apply(self, other = None):
        logging.debug("Loop apply %s %s", self.name, type(other))

        try:
            item = next(self.collection)
            item = item if isinstance(item, tuple) else (item,)

            logging.debug("Loop apply %s %s", self.name, item)

            if self.callback is not None:
                result = self.callback(*item)

            return self.apply(result)

        except StopIteration:
            logging.debug("Loop stop %s %s", self.name, type(other))

        return other
