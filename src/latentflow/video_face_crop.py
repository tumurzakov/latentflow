import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from .flow import Flow
from .video import Video

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from einops import rearrange

class VideoFaceCrop(Flow):
    def __init__(self, cache=None, padding_percent=0.05, resolution=512, zoom=False):
        self.cache = cache
        self.padding_percent = padding_percent
        self.zoom = zoom

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mtcnn = MTCNN(image_size=resolution, device=device)

    def apply(self, video):
        if self.cache is not None and os.path.isfile(self.cache):
            v = torch.load(self.cache).to(video.video.device)
        else:
            v = self.process(video)
            torch.save(v, self.cache)

        return Video('HWC', v)

    @torch.no_grad()
    def process(self, video):
        batches = []
        v = video.hwc()
        for b in v:
            frames = []
            for f in tqdm(b, desc=f'facecrop'):
                boxes, probs, points = self.mtcnn.detect(
                        f.detach().cpu().numpy(),
                        landmarks=True)

                back = torch.zeros_like(f)
                for box in boxes:
                    x0,y0,x1,y1 = [int(x) for x in box]
                    h = y1-y0
                    w = x1-x0
                    ph = int(h * self.padding_percent)
                    pw = int(w * self.padding_percent)
                    x0 = x0 - pw
                    x1 = x1 + pw
                    y0 = y0 - ph
                    y1 = y1 + ph

                    face = f[y0:y1,x0:x1,:]
                    if self.zoom:
                        frame_shape = f.shape
                        face_shape = face.shape
                        scale = min(frame_shape[0]/face_shape[0], frame_shape[1]/face_shape[1])
                        face_chw = rearrange(face, 'h w (b c) -> b c h w', b=1)
                        zoom_face = self.interpolate(face_chw.float(), (scale,scale))
                        zoom_face = rearrange(zoom_face, 'b c h w -> h w (b c)')
                        center_h = back.shape[0]//2
                        center_w = back.shape[1]//2
                        delta_h = min(zoom_face.shape[0],back.shape[0])//2
                        delta_w = min(zoom_face.shape[1],back.shape[1])//2
                        start_h = center_h-delta_h
                        start_w = center_w-delta_w
                        back[
                                start_h:start_h+zoom_face.shape[0],
                                start_w:start_w+zoom_face.shape[1],
                                :
                            ] = zoom_face.to(torch.uint8)

                    else:
                        back[y0:y1,x0:x1,:] = face

                frames.append(torch.tensor(back))

            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)
        batches = batches.to(v.device)
        return batches

    def interpolate(self, tensor, scale_factor):
        return F.interpolate(
                tensor,
                scale_factor = scale_factor,
                mode='bilinear',
                )
