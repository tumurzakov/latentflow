import unittest
from latentflow.video import Video
from latentflow.video_load import VideoLoad

class TestVideo(unittest.TestCase):
    def test_should_load_video(self):
        """
        video|VideoLoad(path="bunny.mp4")
        """

        video = Video()|VideoLoad(path="bunny.mp4")
        self.assertTrue(isinstance(video, Video))
