import torch
import logging
from .flow import Flow
from .latent import Latent

class LatentShrink(Flow):
    def __init__(self, mask, padding=0):
        self.mask = mask
        self.padding = padding

    def apply(self, latent):
        shrink_mask = self.mask.shrink(self.padding)
        logging.debug("LatentShrink %s %s %s", self.mask, shrink_mask, latent)

        latent_tile = shrink_mask.origin_tile
        latent_tile[1] = slice(0, 4)
        shrink_latent = latent[latent_tile]

        logging.debug("LatentShrink %s %s", latent_tile, shrink_latent)

        return shrink_latent

class LatentUnshrink(Flow):
    def __init__(self, latent, mask, padding):
        self.mask = mask
        self.latent = latent
        self.padding = padding

    def apply(self, latent):
        shrink_mask = self.mask.shrink(self.padding)
        logging.debug("LatentUnshrink %s %s %s", self.mask, shrink_mask, latent)

        unshrink_latent = torch.zeros_like(self.latent.latent)
        latent_tile = shrink_mask.origin_tile
        latent_tile[1] = slice(0,4)
        unshrink_latent[latent_tile] = latent.latent
        unshrink_latent = type(latent)(unshrink_latent)

        logging.debug("LatentUnshrink %s", unshrink_latent)

        return unshrink_latent

class VideoShrink(Flow):
    def __init__(self, mask, padding=0):
        self.mask = mask
        self.padding = padding

    def apply(self, video):
        logging.debug("VideoShrink %s %s", self.mask, video)

        shrink_mask = self.mask.shrink(int(self.padding//8))
        video_tile = shrink_mask.origin_tile
        video_tile = (
                slice(0, video.video.shape[0]),
                video_tile[2],
                slice(video_tile[3].start * 8, video_tile[3].stop * 8),
                slice(video_tile[4].start * 8, video_tile[4].stop * 8),
                slice(0, 3),
                )

        shrink_video = video[video_tile]

        logging.debug("VideoShrink %s %s", video_tile, shrink_video)

        return shrink_video
