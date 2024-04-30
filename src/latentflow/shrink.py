import torch
import logging
from .flow import Flow
from .latent import Latent

class LatentShrink(Flow):
    def __init__(self, mask, padding=0):
        self.mask = mask
        self.padding = padding

    def apply(self, latent):
        try:
            shrink_mask = self.mask.shrink(self.padding)
            logging.debug("LatentShrink %s %s %s", self.mask, shrink_mask, latent)

            latent_tile = shrink_mask.origin_tile
            latent_tile[1] = slice(0, 4)
            shrink_latent = latent[latent_tile]

            logging.debug("LatentShrink %s %s", latent_tile, shrink_latent)

            return shrink_latent
        except Exception as e:
            logging.error("LatentUnshrink error %s %s", e, latent)
            raise e

class LatentUnshrink(Flow):
    def __init__(self, latent, mask, padding, displace=False):
        self.mask = mask
        self.latent = latent
        self.padding = padding
        self.displace = displace

    def apply(self, latent):
        shrink_mask = self.mask.shrink(self.padding)
        logging.debug("LatentUnshrink %s %s %s", self.mask, shrink_mask, latent)

        unshrink_latent = torch.zeros_like(self.latent.latent)
        latent_tile = shrink_mask.origin_tile
        latent_tile[1] = slice(0,4)

        latent_tile[3] = slice(latent_tile[3].start, latent_tile[3].start+latent.latent.shape[3])
        latent_tile[4] = slice(latent_tile[4].start, latent_tile[4].start+latent.latent.shape[4])

        if self.displace:
            #TODO: there is displacement on +1 pixel by x and y somewhere in rounding
            latent_tile[3] = slice(latent_tile[3].start-1, latent_tile[3].start-1+latent.latent.shape[3])
            latent_tile[4] = slice(latent_tile[4].start-1, latent_tile[4].start-1+latent.latent.shape[4])

        try:
            unshrink_latent[latent_tile] = latent.latent
            unshrink_latent = type(latent)(unshrink_latent)

            logging.debug("LatentUnshrink %s", unshrink_latent)

            return unshrink_latent
        except Exception as e:
            logging.error("LatentUnshrink error %s %s", e, latent)
            logging.error("LatentUnshrink error %s %s %s %s",
                    latent_tile,
                    unshrink_latent.shape,
                    unshrink_latent[latent_tile].shape,
                    latent.latent.shape)
            raise e

class VideoShrink(Flow):
    def __init__(self, mask, padding=0):
        self.mask = mask
        self.padding = padding

    def apply(self, video):
        logging.debug("VideoShrink %s %s", self.mask, video)
        try:
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
        except Exception as e:
            logging.error("VideoShrink %s %s", e, video)
            raise e

