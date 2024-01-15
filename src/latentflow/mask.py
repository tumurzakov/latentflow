import torch
import logging

import torch.nn.functional as F
from einops import rearrange

from .flow import Flow
from .video import Video

class Mask(Flow):
    def __init__(self, mask: torch.Tensor):
        self.mask = mask
        logging.debug(f'{self}')

    def __str__(self):
        return f'Mask({self.mask.shape})'

class MaskEncode(Flow):
    def __init__(self, threshold = 200):
        self.threshold = threshold
        logging.debug('MaskEncode init')

    def apply(self, video: Video) -> Mask:
        logging.debug('MaskEncode apply %s', video)

        height, width = video.size()

        mask = video.chw()

        masks = []
        for m in mask:
            mask = F.interpolate(
                    m,
                    size=(height//8, width//8),
                    mode='nearest-exact').squeeze()

            mask = mask.sum(dim=1)
            mask = (mask > self.threshold).to(torch.float)
            mask = mask.unsqueeze(0)
            mask = mask.unsqueeze(0)

            masks.append(mask)

        mask = torch.cat(masks)

        mask = Mask(mask=mask)

        logging.debug('MaskEncode mask %s', mask)

        return mask


