import unittest
import torch
import latentflow as lf

class MaskTest(unittest.TestCase):
    def test_should_shrink(self):
        m = torch.tensor(
                [ # b
                    [ # c
                        [ #f
                            [ #hxw
                                [0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            [
                                [0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                            ],
                            [
                                [0, 0, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                            [
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                            ],
                        ],
                    ],
                ],

                dtype=torch.uint8
            )

        mask = lf.Mask(m)
        shrinked = mask.shrink()
        self.assertEqual(shrinked.mask.shape, (1,1,4,2,2))
