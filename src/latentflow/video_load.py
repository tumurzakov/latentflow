import torch
from typing import List, Optional, Tuple, Union, Generator
from pathlib import Path
import decord
decord.bridge.set_bridge('torch')

from .flow import Flow
from .video import Video

class VideoLoad(Flow):

    def __init__(self,
            path: Path,
            start_frame: int = 0,
            video_length: int = None,
            device: Optional[Union[str, torch.device]] = None,
            ):

        self.path = path
        self.start_frame = start_frame
        self.video_length = video_length
        self.device = device

    def apply(self, video: Video) -> Video:
        vr = decord.VideoReader(self.path)
        video_length = self.video_length
        if video_length is None:
            video_length = len(vr)
        sample_index = list(range(self.start_frame, video_length))
        video = vr.get_batch(sample_index)

        if self.device is not None:
            video = video.to(self.device)

        return Video('HWC', video=video)

