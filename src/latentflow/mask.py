import torch
import logging

import torch.nn.functional as F
from einops import rearrange

import cv2
import numpy as np

from .flow import Flow
from .video import Video
from .video_show import VideoShow
from .latent import Latent

class Mask(Flow):
    def __init__(self,
            mask: torch.Tensor,
            onload_device: str='cuda',
            offload_device: str='cpu',
            mode: str = 'b c f h w',
            origin = None,
            origin_tile = None,
            threshold = 200,
            ):
        self.mask = mask
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.origin = origin
        self.origin_tile = origin_tile
        self.mode = mode
        logging.debug(f'{self}')

    def onload(self):
        self.mask = self.mask.to(self.onload_device)

    def offload(self):
        self.mask = self.mask.to(self.offload_device)

    def binarize(self, threshold=0.5):
        return Mask(torch.clamp((self.mask > threshold).to(torch.float), 0, 1), origin=self)

    def apply(self, other):
        self.onload()
        other.onload()

        result = other
        if isinstance(other, Latent):
            result = Latent(other.latent * self.mask)
        elif isinstance(other, Video):
            result = Video('HWC', other.video * self.mask)

        other.offload()
        self.offload()
        result.offload()

        return result

    def cfg(self, guidance_scale):
        repeat = 2 if guidance_scale > 1.0 else 1
        return Mask(self.mask.repeat(repeat,1,1,1,1))

    def __str__(self):
        return f'Mask({self.mask.shape})'

    def __len__(self):
        return self.mask.shape[2]

    def __getitem__(self, key):
        return Mask(self.mask[key], origin=self, origin_tile=key)

    def __add__(self, other):
        return Mask(torch.clamp(self.mask + other.mask, 0, 1))

    def __sub__(self, other):
        if isinstance(other, Mask):
            return Mask(torch.clamp(self.mask - other.mask, 0, 1))
        return self

    def size(self):
        return (self.mask.shape[2], self.mask.shape[3])

    def invert(self):
        return Mask(1-self.mask)

    def video(self):
        if self.mode == 'b f h w c':
            return Video('HWC', self.mask)
        else: # latent
            mask = self.mask.float()
            mask = F.interpolate(
                mask,
                size=(mask.shape[2], mask.shape[3]*8, mask.shape[4]*8),
                mode='trilinear',
                align_corners=False
                )
            mask = rearrange(mask, f'{self.mode} -> b f h w c')
            mask = mask.repeat(1,1,1,1,3)
            mask = mask * 255
            mask = mask.to(torch.uint8)

            return Video('HWC', mask)

    def hwc(self):
        return self.video().hwc()

    def resize(self, size):
        v = self.mask
        v = F.interpolate(
                v,
                size=size,
                mode='trilinear',
                align_corners=False
                )
        return Mask(mask=v)

    def flatten(self):
        orig_mask = mask = self.mask
        if self.mode != 'b f c h w':
            mask = rearrange(mask, f'{self.mode} -> b f c h w')

        mask = torch.sum(mask, dim=2, keepdim=True) # (1, f, 1, h, w)
        mask = torch.sum(mask, dim=1, keepdim=True) # (1, 1, 1, h, w)
        mask = mask.clamp(0, 1)
        mask = mask.to(torch.uint8)
        mask = mask.repeat(1, orig_mask.shape[1], orig_mask.shape[2], 1, 1)

        tile =[
                slice(0, mask.shape[0]),
                slice(0, mask.shape[1]),
                slice(0, mask.shape[2]),
                slice(0, mask.shape[3]),
                slice(0, mask.shape[4]),
                ]

        return Mask(mask, origin=self, origin_tile=tile)

    def shrink(self, padding=0):
        mask = self.mask
        if self.mode != 'b f c h w':
            mask = rearrange(mask, f'{self.mode} -> b f c h w')

        mask = torch.sum(mask, dim=2, keepdim=True) # (1, f, 1, h, w)
        mask = torch.sum(mask, dim=1, keepdim=True) # (1, 1, 1, h, w)
        mask = mask.clamp(0, 1)
        mask = mask.to(torch.uint8)

        chan = mask[0][0][0]
        chan = chan.detach().cpu().numpy()
        contours, _ = cv2.findContours(chan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape = self.mask.shape
        tile =[
                slice(0, shape[0]),
                slice(0, shape[1]),
                slice(0, shape[2]),
                slice(0, shape[3]),
                slice(0, shape[4]),
                ]

        origin = None
        origin_tile = tile
        if contours:
            merged_contour = contours[0]
            for contour in contours[1:]:
                merged_contour = np.concatenate((merged_contour, contour))

            x, y, w, h = cv2.boundingRect(merged_contour)

            if w>0 and h>0:
                dims = self.mode.split(' ')
                tile[dims.index('h')] = slice(max(0, y-padding), max(0, y+h+padding))
                tile[dims.index('w')] = slice(max(0, x-padding), max(0, x+w+padding))
                origin = self
                origin_tile = tile

        return Mask(self.mask[tile], origin=origin, origin_tile=origin_tile)

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

        mask = Mask(mask=mask, mode=self.mode)

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

class MaskShow(Flow):
    def __init__(self, fps):
        self.fps = fps

    def apply(self, mask):
        video = mask.video()
        VideoShow(fps=self.fps).apply(video)
        return mask

class MaskFlatten(Flow):
    def apply(self, other):
        assert isinstance(other, Mask)
        return Mask(other.mask.flatten())

class MaskGrow(Flow):

    def __init__(self,
            transparent=1.0,
            kernel_size=3,
            stride=1,
            padding=1,
            ):
        self.transparent = transparent
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding = padding

    def apply(self, mask):
        shape = mask.mask.shape
        m = mask.mask.clone().to(torch.float)
        diffs = torch.zeros_like(m)
        for f in range(shape[2]):
            frame = m[:,:,f,:,:]
            pool = F.max_pool2d(frame,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding)
            diff =  pool - frame
            m[:,:,f,:,:] += diff*self.transparent

            diffs[:,:,f,:,:] += diff

        return Mask(m.clamp(0,1))

class MaskBlur(Flow):
    def apply(self, mask):

        conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        blur_kernel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32) / 16.0
        blur_kernel = blur_kernel.view(1, 1, 3, 3)  # Reshape to match the shape of the convolutional kernel
        conv.weight.data.copy_(blur_kernel)

        shape = mask.mask.shape
        m = mask.mask.clone().to(torch.float)
        for f in range(shape[2]):
            m[:,:,f,...] = conv(m[:,:,f,...])

        return Mask(m.clamp(0,1))


class MaskMergeWithSlidingWindow(Flow):
    def __init__(self, window):
        self.window = window

    def apply(self, mask):
        # b c f h w

        shape = mask.mask.shape
        window = self.window

        m = mask.mask.clone().to(torch.float)
        for f in range(shape[2]-1, window-1, -1):
            for i in range(window):
                m[:,:,f,:,:] += m[:,:,f-i,:,:]

        return Mask(m.clamp(0,1))

