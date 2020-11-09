import numpy as np
import cv2
import os
from src.types.parent_slide import SlideImage

class FluorescenceSlide(SlideImage):

    def __init__(self, img, **kwargs):
        """
        Arguments:
            img (ndarray): uint16 array of shape (h,w,3)
        """
        self.img = img
        self.height, self.width, self.channels = self.img.shape
        assert self.channels == 3, f'FluorescenceSlide must have 3 channels but found {self.channels}'

        self.name = kwargs['name'] if 'name' in kwargs.keys() else ''

    @classmethod
    def fromFluorescenceSlides(cls, slides:list, weights:list=None, **kwargs):
        if len(slides) == 1:
            return cls(slides[0].img, **kwargs)
        
        stack = np.stack([slide.img for slide in slides])
        avg = np.average(stack, axis=0)
    
        return cls(avg, **kwargs)
    
    def write_to(self, out_dir):
        """
        Write three channels of the FluorescenceSlide as .tif files to a specified directory
        """
        assert self.name is not None, 'name is missing from FluorescenceSlide'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        for chan in range(3):
            chan_img = self.img[:,:,chan].astype(np.uint16)
            file_suffix = f'A0{chan+1}Z01C0{chan+1}.tif'
            fn = os.path.join(out_dir, self.name + file_suffix)
            cv2.imwrite(fn, chan_img)
        