import random
import numpy as np
from enum import Enum
from fastai.vision.all import *
from src.types.parent_slide import TileTypes
from src.types.tensor_tiles import BrightfieldTile, FluorescenceTile
import torch.nn.functional as nnf

class AugmentationOptions(Enum):
    flip_h = 1
    flip_v = 2
    contrast = 3
    color = 4
    crop = 5

class PairAugmentations:
    """
    Apply augmentations for Brightfield and Fluorescence tiles.
    Flip augmentations are applied for both images. 
    """
    
    def __init__(
        self, 
        flip_p=0.5, 
        lightness_p=0.5, 
        contrast_lim=0.2,
        color_p=1.0,
        color_lim=0.5,
        crop_p=0.5,
        crop_sz_p=0.8
        ):
        self.flip_p = flip_p
        self.lightness_p = lightness_p
        self.contrast_lim = contrast_lim
        self.color_p = color_p
        self.color_lim = color_lim
        self.crop_p = crop_p
        self.crop_sz_p = crop_sz_p
        
    def _flip_h(self, img:TensorImage):
        return torch.flip(img, [2])
    
    def _flip_v(self, img:TensorImage):
        return torch.flip(img, [1])
    
    def _contrast(self, img:TensorImage):
        contrast = random.random() * self.contrast_lim
        if random.random() < 0.5: contrast *= -1.0
        return img * (1.0 + contrast)

    def _color(self, img:TensorImage):
        c, _, _ = img.shape
        color_factor = 1 + (np.random.rand(c) - 0.5) * self.color_lim * 2.0
        return torch.mul(img.permute(1,2,0), tensor(color_factor)).permute(2,0,1)
    
    def _crop(self, img:TensorImage):
        _, h, w = img.shape
        crop_sz = int(h * self.crop_sz_p)
        start_x = int((w - crop_sz - 1) * random.random())
        start_y = int((h - crop_sz - 1) * random.random())
        img = img[:, start_x:start_x + crop_sz, start_y:start_y + crop_sz]
        return nnf.interpolate(
            img.unsqueeze(0), 
            size=(h, w), 
            mode='bicubic', 
            align_corners=False)[0]

    def _brightfield_transform(self, img:TensorImage, augs:list):
        """ Flips and contrast """
        if AugmentationOptions.flip_h in augs:
            img = self._flip_h(img)
        if AugmentationOptions.flip_v in augs:
            img = self._flip_v(img)
        if AugmentationOptions.contrast in augs:
            img = self._contrast(img)
        return BrightfieldTile(img)
        
    def _fluorescence_transform(self, img:TensorImage, augs:list):
        """ Flips """
        if AugmentationOptions.flip_h in augs:
            img = self._flip_h(img)
        if AugmentationOptions.flip_v in augs:
            img = self._flip_v(img)
        return FluorescenceTile(img)

    def _fluorescence_contrastive_transform(self, img:TensorImage):
        """ Stronger augmentations """
        # re-generate augmentations
        augs = self._get_augs()
        if AugmentationOptions.flip_h in augs:
            img = self._flip_h(img)
        if AugmentationOptions.flip_v in augs:
            img = self._flip_v(img)
        if AugmentationOptions.contrast in augs:
            img = self._contrast(img)
        if AugmentationOptions.color in augs:
            img = self._color(img)
        if AugmentationOptions.crop in augs:
            img = self._crop(img)
        return FluorescenceTile(img)
        
    def _get_augs(self):
        augs = []
        augs += [AugmentationOptions.flip_h] if random.random() < self.flip_p else []
        augs += [AugmentationOptions.flip_v] if random.random() < self.flip_p else []
        augs += [AugmentationOptions.contrast] if random.random() < self.lightness_p else []
        augs += [AugmentationOptions.color] if random.random() < self.color_p else []
        augs += [AugmentationOptions.crop] if random.random() < self.crop_p else []
        return augs
    
    def __call__(self, data:dict):
        augs = self._get_augs()

        if TileTypes.brightfield in data:
            data[TileTypes.brightfield] = self._brightfield_transform(data[TileTypes.brightfield], augs)
            
        if TileTypes.fluorescence in data:
            data[TileTypes.fluorescence] = self._fluorescence_transform(data[TileTypes.fluorescence], augs)

        if TileTypes.fluorescence_contrastive in data:
            data[TileTypes.fluorescence_contrastive] = self._fluorescence_contrastive_transform(data[TileTypes.fluorescence_contrastive])
        
        return data