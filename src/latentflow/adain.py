import torch
import logging

from .flow import Flow

# AdaIN https://github.com/naoto0804/pytorch-AdaIN
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)
    N, C, F = size[:3]
    feat_var = feat.view(N, C, F, -1).var(dim=3) + eps
    feat_std = feat_var.sqrt().view(N, C, F, 1, 1)
    feat_mean = feat.view(N, C, F, -1).mean(dim=3).view(N, C, F, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:3] == style_feat.size()[:3])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class Adain(Flow):
    def __init__(self, style):
        self.style = style

    def apply(self, video):
        batches = []
        for b in video.hwc():
            frames = []
            for f in b:
                frame = adaptive_instance_normalization(f, self.style)
                frames.append(frame)
            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)

        return Video('HWC', batches)
