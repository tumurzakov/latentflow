import torch
import logging
from PIL import Image
import numpy as np
from einops import rearrange

from .flow import Flow
from .video import Video
from .video_load import VideoLoad
from .interpolate import Interpolate
from .tensor import Tensor
from .video_show import VideoShow

# AdaIN https://github.com/naoto0804/pytorch-AdaIN
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5), f"Must have 5 dims, but have {len(size)}"
    N, C, F = size[:3]
    feat_var = feat.view(N, C, F, -1).var(dim=3) + eps
    feat_std = feat_var.sqrt().view(N, C, F, 1, 1)
    feat_mean = feat.view(N, C, F, -1).mean(dim=3).view(N, C, F, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:3] == style_feat.size()[:3]), \
        f'Size must be same {content_feat.size()}!={style_feat.size()}'

    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class Adain(Flow):
    def __init__(self, style):
        self.style = style

        if isinstance(style, str):

            if '.png' in self.style:
                self.style = torch.tensor(np.array(Image.open(self.style)))
            elif '.mp4' in latents_adain_path:
                self.style = VideoLoad(self.style).apply()

    def apply(self, video):

        vid = video.hwc()
        video_length = vid.shape[1]

        ref = self.style
        if len(ref.shape) == 3: #image
            ref = ref.unsqueeze(0).unsqueeze(0)
        elif len(ref.shape) == 4: #video
            ref = ref.unsqueeze(0)
        ref = ref.repeat(1,video_length,1,1,1)
        ref = rearrange(ref, 'b f h w c -> b c f h w')
        ref = ref.to(vid.device, torch.float)/255
        ref = Interpolate(size=(video_length, vid.shape[2], vid.shape[3]), mode='trilinear').apply(Tensor(ref)).tensor

        vid = vid.to(torch.float)/255
        vid = rearrange(vid, 'b f h w c -> b c f h w')
        vid = adaptive_instance_normalization(vid, ref)
        vid = torch.clamp(vid * 255, 0, 255).to(torch.uint8)
        vid = rearrange(vid, 'b c f h w -> b f h w c')

        return Video('HWC', vid)
