import unittest
import torch
import latentflow as lf

class MaskTest(unittest.TestCase):
    def test_should_shrink(self):
        m = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ], dtype=torch.uint8
            ).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        mask = lf.Mask(m)
        shrinked = mask.shrink()
        self.assertEqual(shrinked.mask.shape, (1,1,1,1,1))
