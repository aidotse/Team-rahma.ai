import numpy as np
import cv2
from src.types.parent_slide import SlideImage

class BrightFieldSlide(SlideImage):

    def __init__(self, img, **kwargs):
        """
        Arguments:
            img (ndarray): uint16 array of shape (h,w,c)
        """
        self.img = img
        self.height, self.width, self.channels = self.img.shape
        assert self.channels == 7, f'BrightFieldSlide must have 7 channels but found {self.channels}'
        self.name = kwargs['name'] if 'name' in kwargs.keys() else ''