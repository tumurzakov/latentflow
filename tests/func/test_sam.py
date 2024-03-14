import os
import unittest
import latentflow as lf

class TestSAM(unittest.TestCase):
    def test_should_segment(self):
        models = os.getenv('MODELS_DIR')

        state = lf.State({})

        lf.LoadSegmentAnything.load(model_type='vit_h', model_path=f'{models}/sam_vit_h_4b8939.pth') | lf.Set(state, 'sam')

        (lf.VideoLoad(['assets/bunny.mp4'], video_length=4, width=512, height=288).apply()
            | lf.SegmentAnything(sam_model=state['sam'])
            | lf.Debug("Mask")
            | lf.MaskShow(fps=16)
            )

