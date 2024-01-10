import logging
import torch
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow

class Loop(Flow):

    def __init__(self,
            collection:List = [],
            callback:Optional[Callable] = None,
            ):
        self.collection = collection
        self.current_item = 0
        self.callback = callback

        logging.debug("Loop init")

    def apply(self, other = None):

        try:
            index, item = next(self.collection)

            logging.debug("Loop apply %s %s", index, item)

            if self.callback is not None:
                result = self.callback(index, item)

            return self.apply(result)

        except StopIteration:
            logging.debug("Loop stop")

        return other
