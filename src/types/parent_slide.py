import numpy as np
import cv2
import os
from enum import Enum

def _padIfNeeded(img, target_width: int = 128, target_height: int = 128):
    ''' Pad images that need padding (padding on right and bottom)'''

    h, w, c = img.shape
    pad_right = max(0,target_width-w)
    pad_bottom = max(0,target_height-h)
    if pad_right>0 or pad_bottom>0:
        img = cv2.copyMakeBorder(img, 0,  pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
    return img

def _getOverlapMask(top_left_corners:list, tile_sz:int):
    # make an empty mask that corresponds to the original image size
    xmax = np.array(top_left_corners)[...,0].max(); ymax = np.array(top_left_corners)[...,1].max()
    mask = np.zeros((ymax + tile_sz, xmax + tile_sz))
    # iterate through the tile coords and cumulately sum with +1:
    # as result, the overlapping areas will have pixel values >1
    for x, y in top_left_corners:
        mask[y:y + tile_sz, x:x + tile_sz] += 1
    # map the overlap values to 0=no overlap, 1=2 overlapping tiles and 4=4 overlapping tiles
    mask[np.where(mask <= 2)] += -1
    return mask

class TileTypes(Enum):
    brightfield = 1
    fluorescence = 2
    fluorescence_contrastive = 3

class SlideImage:
    """
    SlideImage holds full (non-tile) fluorescent or brightfield images and contains all channels in one object.
    
    This is a parent class for 
    - FluorescenceSlide (brightfield_slide.py)
    - BrightFieldSlide (fluorescence_slide.py)
    """

    def __init__(self, img, **kwargs):
        """
        Arguments:
            img (ndarray): uint16 array of shape (h,w,c)
        """
        self.img = img
        self.height, self.width, self.channels = self.img.shape
        self.name = kwargs['name'] if 'name' in kwargs.keys() else ''

    @classmethod
    def fromFiles(cls, fns:list, **kwargs):
        """
        Creates a SlideImage from several channel files.

        Arguments:
            fns (list:str): paths to channels files

        Returns:
            SlideImage
        """
        
        # sort to make sure the files are in order
        fns.sort()
        
        img = np.stack([cv2.imread(fn, cv2.IMREAD_UNCHANGED) for fn in fns], axis=2)
        return cls(img, **kwargs)

    @classmethod
    def fromTiles(cls, tiles:list, **kwargs):
        """
        Stitch a SlideImage from SlideTile-list.
        """

        # sort to make sure the files are in order
        top_left_corners = np.array([(tile.start_x, tile.start_y) for tile in tiles])
        tile_sz = tiles[0].tile_sz; overlap = tiles[0].overlap
        orig_w = tiles[0].orig_w; orig_h = tiles[0].orig_h
        ch = tiles[0].img.shape[-1]; dtype = tiles[0].img.dtype
        # make an empty image that corresponds to the original image size
        xmax = top_left_corners[..., 0].max(); ymax = top_left_corners[..., 1].max()
        stitched_img = np.zeros((ymax+tile_sz, xmax+tile_sz, ch), dtype=np.float64)
        # get the overlap mask
        overlap_mask = _getOverlapMask(top_left_corners, tile_sz)
        grad = np.linspace(0, 1, overlap)
        # for each tile calculate the alpha according to which to weighted sum the tile values
        for tile, (x, y) in zip(tiles, top_left_corners):
            tile = tile.img
            tile_overlap_mask = overlap_mask.copy()[y:y+tile_sz, x:x+tile_sz]
            alpha = np.zeros(tile.shape[:2])
            # top margin
            if not np.any(tile_overlap_mask[:overlap, ...] == 0):
                alpha[:overlap, ...] += np.divide(grad[..., None], tile_overlap_mask[:overlap, ...])
            # bottom margin
            if not np.any(tile_overlap_mask[-overlap:, ...] == 0):
                alpha[-overlap:, ...] += np.divide(grad[::-1][..., None], tile_overlap_mask[-overlap:, ...])
            # left margin
            if not np.any(tile_overlap_mask[..., :overlap] == 0):
                alpha[..., :overlap] += np.divide(grad[None, ...], tile_overlap_mask[..., :overlap])
            # right margin
            if not np.any(tile_overlap_mask[..., -overlap:] == 0):
                alpha[..., -overlap:] += np.divide(grad[::-1][None, ...], tile_overlap_mask[..., -overlap:])
            # areas with no overlap are to be summed with alpha value 1
            alpha[np.where(tile_overlap_mask == 0)] = 1
            # add the weighted tile the background
            stitched_img[y:y+tile_sz, x:x+tile_sz] += np.multiply(tile, alpha[..., None])

        # return the original size and dtype
        stitched_img = stitched_img[:orig_h, :orig_w]
        stitched_img = stitched_img.astype(dtype)
        return cls(stitched_img, **kwargs)

    def getTiles(self, tile_sz:int=128, overlap:int=32):

        assert tile_sz//2>overlap, "`overlap` should be strictly less than half of `tile_sz`"
        orig_h, orig_w, _ = self.img.shape
        tiles = []
        rolling_x = 0; rolling_y = 0
        while rolling_x <= self.width:
            while rolling_y <= self.height:
                xmin = rolling_x
                xmax = rolling_x + tile_sz
                ymin = rolling_y
                ymax = rolling_y + tile_sz
                tile = self.img[ymin:ymax, xmin:xmax]
                tile = _padIfNeeded(tile, tile_sz, tile_sz)
                tiles.append(SlideTile(tile, xmin, ymin, tile_sz, overlap, orig_w, orig_h, self.name))
                rolling_y += tile_sz - overlap
            rolling_x += tile_sz - overlap
            rolling_y = 0
        return tiles

class SlideTile:
    """
    SlideTile is used for storing tile images to disk.
    Images are saved as uint16 numpy arrays.
    Slide name and position is encoded to file name. 
    """
    
    def __init__(self, img, start_x:int, start_y:int, tile_sz:int, overlap:int,
                 orig_w:int, orig_h:int, name:str):
        self.img = img
        self.start_x = start_x
        self.start_y = start_y
        self.tile_sz = tile_sz
        self.overlap = overlap
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.name = name

    def write_file(self, dir, slide_name=None):
        """
        Write SlideTile to a file. It is saved as numpy array.
        File name: {slide_name}__[start_x}__{start_y}.npy

        Arguments:
            dir        (str): save directory
            slide_name (str): Optional name for slide. If None (default), slide name is retrieved from parent slide.
        
        Returns:
            fn         (str): file path to written tile
        """
        assert self.name != '', 'Slide does not have a name'
        fn = os.path.join(dir, self.name + f'__{self.start_x}__{self.start_y}.npy')
        np.save(fn, self.img)
        
        return fn