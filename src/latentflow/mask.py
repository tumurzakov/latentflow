import torch
import logging

import torch.nn.functional as F
from einops import rearrange

from .flow import Flow
from .video import Video
from .latent import Latent

class Mask(Flow):
    def __init__(self,
            mask: torch.Tensor,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):
        self.mask = mask
        self.onload_device = onload_device
        self.offload_device = offload_device
        logging.debug(f'{self}')

    def onload(self):
        self.mask = self.mask.to(self.onload_device)

    def offload(self):
        self.mask = self.mask.to(self.offload_device)

    def __str__(self):
        return f'Mask({self.mask.shape})'

    def __getitem__(self, key):
        return Mask(self.mask[key])

    def size(self):
        return (self.mask.shape[2], self.mask.shape[3])

    def invert(self):
        return Mask(1-self.mask)

    def video(self):
        return Video('HWC', self.mask)

    def resize(self, size):
        v = self.mask
        v = F.interpolate(
                v,
                size=size,
                mode='trilinear',
                align_corners=False
                )
        return Mask(mask=v)


class MaskEncode(Flow):
    def __init__(self,
            width = None,
            height = None,
            threshold = 200,
            channels=1,
            mode='b c f h w',
            ):

        self.width = width
        self.height = height
        self.threshold = threshold
        self.channels = channels
        self.mode = mode
        logging.debug('MaskEncode init')

    def apply(self, video: Video) -> Mask:
        logging.debug('MaskEncode apply %s', video)

        video.onload()

        mask = video.chw()

        if self.width is None or self.height is None:
            self.height, self.width = video.size()

        mask = F.interpolate(
                mask,
                size=(1, self.height, self.width),
                mode='nearest-exact').squeeze()

        mask = (mask > self.threshold).to(torch.uint8)
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask = mask.repeat(self.channels, 1, 1, 1, 1)
        mask = rearrange(mask, f'c b f h w -> {self.mode}')

        mask = Mask(mask=mask)

        logging.debug('MaskEncode mask %s', mask)

        video.offload()
        mask.offload()

        return mask

class LatentMaskCrop(Flow):
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.mask = None
        self.mask_crop = None
        self.crops = []

    @torch.no_grad()
    def restore(self, latent, mode='bilinear'):

        l = latent.latent
        m = self.mask.mask
        back = torch.zeros((1,4,m.shape[3],m.shape[4])).to(l.device)

        frames = []
        for f in range(l.shape[2]):
            frame = l[:,:,f,:,:]
            frame_crop = self.crops[f]
            mask_crop = self.mask_crop.mask[:,:,f,:,:]

            frame = frame * mask_crop
            crop = frame[frame_crop['dst_slice']]

            scale = 1/frame_crop['scale']

            c = F.interpolate(
                    crop,
                    scale_factor = scale,
                    mode=mode,
                    )

            b = back.clone()
            bc = back[frame_crop['src_slice']]

            bc[
                0:min(c.shape[0], b.shape[0]),
                0:min(c.shape[1], b.shape[1]),
                0:min(c.shape[2], b.shape[2]),
                0:min(c.shape[3], b.shape[3]),
            ] = c[
                0:min(c.shape[0], b.shape[0]),
                0:min(c.shape[1], b.shape[1]),
                0:min(c.shape[2], b.shape[2]),
                0:min(c.shape[3], b.shape[3]),
            ]

            frames.append(b)

        frames = torch.stack(frames, dim=2)

        l = F.interpolate(
                frames,
                size=(l.shape[2], self.height, self.width),
                mode='trilinear',
                align_corners=False
                )

        return Latent(l)

    def apply(self, mask: Mask):
        self.mask = mask

        m = mask.mask
        shape = m.shape

        back = torch.zeros((1,4,self.height,self.width)).to(m.device)

        frames = []
        for f in range(shape[2]):
            frame = m[:,:,f,:,:]

            # Step 2: Find the bounding box of the non-empty region
            non_empty_indices = torch.nonzero(frame)
            min_indices, _ = torch.min(non_empty_indices, dim=0)
            max_indices, _ = torch.max(non_empty_indices, dim=0)

            # Bounding box coordinates
            min_h, min_w = min_indices[2], min_indices[3]
            max_h, max_w = max_indices[2] + 1, max_indices[3] + 1

            src_slice = (
                    slice(0, shape[0]),
                    slice(0, shape[1]),
                    slice(min_h.item(), max_h.item()),
                    slice(min_w.item(), max_w.item()),
                    )


            crop = frame[src_slice]

            scale_h = back.shape[2]/crop.shape[2]
            scale_w = back.shape[3]/crop.shape[3]
            scale = min(scale_h, scale_w)

            crop = F.interpolate(
                    crop,
                    scale_factor = scale,
                    mode='bilinear',
                    align_corners=False
                    )

            b = back.clone()

            dst_slice = (
                    slice(0, back.shape[0]),
                    slice(0, back.shape[1]),
                    slice(0, crop.shape[2]),
                    slice(0, crop.shape[3]),
                    )

            b[dst_slice] = crop
            frames.append(b)

            self.crops.append({
                'src_slice': src_slice,
                'dst_slice': dst_slice,
                'scale': scale,
                })

        m = torch.stack(frames, dim=2)

        self.mask_crop = Mask(m)

        return self

class VideoMaskCut(Flow):
    def __init__(self, mask, width, height):
        self.mask = mask
        self.width = width
        self.height = height

    def apply(self, video):

        v = video.chw().float()

        shape = v.shape

        m = F.interpolate(
                self.mask.mask,
                size=(shape[1], shape[3], shape[4]),
                mode='trilinear',
                align_corners=False
                )
        m = rearrange(m, 'b c f h w -> b f c h w')

        v = v * m

        back = torch.zeros((1,3,self.height,self.width)).to(v.device)

        frames = []
        for f in range(shape[1]):
            frame = v[:,f,:,:,:]
            mframe = m[:,f,:,:,:]

            # Step 2: Find the bounding box of the non-empty region
            non_empty_indices = torch.nonzero(mframe)
            min_indices, _ = torch.min(non_empty_indices, dim=0)
            max_indices, _ = torch.max(non_empty_indices, dim=0)

            # Bounding box coordinates
            min_h, min_w = min_indices[2], min_indices[3]
            max_h, max_w = max_indices[2] + 1, max_indices[3] + 1

            crop = frame[:,:,min_h:max_h,min_w:max_w]

            scale_h = back.shape[2]/crop.shape[2]
            scale_w = back.shape[3]/crop.shape[3]
            scale = min(scale_h, scale_w)

            crop = F.interpolate(
                    crop,
                    scale_factor = scale,
                    mode='bilinear',
                    align_corners=False
                    )

            b = back.clone()
            b[:, :, 0:crop.shape[2], 0:crop.shape[3]] = crop
            frames.append(b)

        v = torch.stack(frames)
        v = rearrange(v, 'f b c h w -> b f c h w')

        return Video('CHW', video=v.to(torch.uint8))


class LatentMaskCut(Flow):
    def __init__(self, latent_mask_crop=None):
        self.width = latent_mask_crop.width
        self.height = latent_mask_crop.height
        self.latent_mask_crop = latent_mask_crop

    def apply(self, latent):

        shape = latent.latent.shape

        l = latent.latent
        m = self.latent_mask_crop.mask.mask

        l = l * m

        back = torch.zeros((1,4,self.height,self.width)).to(l.device)

        frames = []
        for f in range(shape[2]):
            frame = l[:,:,f,:,:]

            frame_crop = self.latent_mask_crop.crops[f]

            crop = F.interpolate(
                    frame[frame_crop['src_slice']],
                    scale_factor = frame_crop['scale'],
                    mode='bilinear',
                    align_corners=False
                    )

            b = back.clone()
            b[:, :, 0:crop.shape[2], 0:crop.shape[3]] = crop
            frames.append(b)

        l = torch.stack(frames, dim=2)

        return Latent(latent=l)

class LatentMaskMerge(Flow):
    def __init__(self, background: Latent, foreground:Latent, mask: Mask = None):
        self.background = background
        self.foreground = foreground

        self.mask = mask
        if self.mask is None:
            self.mask = Mask(torch.ones_like(self.latent.latent))

    def apply(self, other):
        return Latent(self.background.latent * (1 - self.mask.mask) + self.foreground.latent * self.mask.mask)
