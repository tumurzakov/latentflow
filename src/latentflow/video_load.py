import torch
import logging
from typing import List, Optional, Tuple, Union, Generator
from pathlib import Path
import decord
decord.bridge.set_bridge('torch')

from .flow import Flow
from .video import Video

class VideoLoad(Flow):

    def __init__(self,
            path: Union[Path, List[Path]],
            start_frame: int = 0,
            video_length: int = None,
            device: Optional[Union[str, torch.device]] = None,
            width: int = -1,
            height: int = -1,
            ):

        self.path = path if isinstance(path, list) else [path]
        self.start_frame = start_frame
        self.video_length = video_length
        self.device = device
        self.width = width
        self.height = height

        logging.debug('VideoLoad init %s', path)

    def apply(self, other=None) -> Video:
        logging.debug('VideoLoad apply %s', self.path)

        path = self.path
        if not isinstance(path, list):
            path = [path]

        videos = []
        for p in path:
            vr = decord.VideoReader(p, width=self.width, height=self.height)
            video_length = self.video_length
            if video_length is None:
                video_length = len(vr)
            sample_index = list(range(self.start_frame, self.start_frame + video_length))
            video = vr.get_batch(sample_index)

            if self.device is not None:
                video = video.to(self.device)

            videos.append(video)

        videos = torch.stack(videos)

        video = Video('HWC', video=videos)

        logging.debug('VideoLoad apply %s', video)

        return video


