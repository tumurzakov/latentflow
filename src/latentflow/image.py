import torch
import logging
import PIL
import numpy as np
import os

from .flow import Flow

class LoadImage(Flow):
    def __init__(self, path=[], image=None, size=None, device=None):
        self.path = path
        self.image = image

        if len(self.path) > 0:
            image = []
            for path in self.path:
                assert os.path.isfile(path), f"File {path} must exist"
                img = PIL.Image.open(path)
                image.append(torch.tensor(np.array(img)))
            image = torch.stack(image)
            if device:
                image = image.to(device)
            self.image = image

        logging.debug(f'{self}')

    def __str__(self):
        return f'Image({self.image.shape})'

    def apply(self, other):
        return other
