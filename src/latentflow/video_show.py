import logging
import mediapy
import numpy as np

from .flow import Flow
from .video import Video

class VideoShow(Flow):
    def __init__(self, fps=25):
        self.fps = fps

    def apply(self, video) -> Video:
        logging.debug('ShowVideo(%s)', video)

        videos = video.hwc().detach().cpu()
        for v in videos:
            frames = [np.array(img) for img in v]
            mediapy.show_video(frames, fps=self.fps)

        return video

