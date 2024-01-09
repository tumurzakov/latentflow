import torch

class Tile:
    def __init__(self,
            width: int,
            height: int,
            length: int,
            width_overlap:int=0,
            height_overlap:int=0,
            length_overlap:int=0):

        self.width = width
        self.height = height
        self.length = length
        self.width_overlap = width_overlap
        self.height_overlap = height_overlap
        self.length_overlap = length_overlap
