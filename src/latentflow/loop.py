import logging
import torch
from typing import Callable, List, Optional, Tuple, Union, Generator
from tqdm import tqdm

from .flow import Flow

class Loop(Flow):

    def __init__(self,
            collection,
            callback:Optional[Callable] = None,
            name: Optional[str] = "",
            progress_bar: bool = False,
            ):

        self.collection = collection
        if isinstance(self.collection, list):
            self.collection = iter(self.collection)

        self.callback = callback
        self.name = name
        self.index = 0

        self.progress_bar = None
        if progress_bar:
            self.progress_bar = tqdm(total=len(self.collection), desc=self.name)

        logging.debug("Loop init %s %s", self.name, type(self.collection))

    def apply(self, other = None):
        logging.debug("Loop apply %s %s", self.name, type(other))

        try:
            item = next(self.collection)

            logging.debug("Loop apply %s %s %s", self.name, self.index, item)

            if self.callback is not None:
                result = self.callback(self.index, item)

            self.index += 1
            if self.progress_bar is not None:
                self.progress_bar.update()

            return self.apply(result)

        except StopIteration:
            logging.debug("Loop stop %s %s", self.name, type(other))

        return other
