# BrightfieldTile and FluorescenceTile subclass TensorImage
# they are CNN input and output data types.

from fastai.vision.all import *
import torch
from src.stats_reader import DataStats

UINT16_MAX = (256**2 - 1)

def _numpy_normalize_brightfield(np_img, stats:DataStats):
    im = TensorImage(np_img.astype(np.float32))
    im = (im - tensor(stats.BRIGHTFIELD_MEAN)) / tensor(stats.BRIGHTFIELD_STD) 
    return im.permute(2,0,1)

class BrightfieldTile(Tensor):

    @classmethod
    def create(cls, fn, stats:DataStats):
        return cls(_numpy_normalize_brightfield(np.load(fn), stats))
    
    @classmethod
    def from_numpy(cls, np_img, stats:DataStats):
        return cls(_numpy_normalize_brightfield(np_img, stats))
    
    def show(self, ctx=None, **kwargs): 
        img = self
        if not isinstance(img, Tensor):
            t = tensor(img)
            t = t.permute(2,0,1)
        else: 
            t = img    
        t = t.permute(1,2,0)
        
        # denormalize
        assert 'stats' in kwargs.keys(), 'stats missing from show'
        stats = kwargs['stats']
        kwargs.pop('stats')
        t = (t * tensor(stats.BRIGHTFIELD_STD)) + tensor(stats.BRIGHTFIELD_MEAN)

        # transform to 0-1 range
        t = t / UINT16_MAX
        
        # convert to 3D PCA - these are pca components fractions for each brightfield channel
        h,w,c = t.shape
        t = t.view(-1,c).float()
        
        t = torch.stack([
            torch.matmul(t, tensor(stats.BRIGHTFIELD_PCA_0)),
            torch.matmul(t, tensor(stats.BRIGHTFIELD_PCA_1)),
            torch.matmul(t, tensor(stats.BRIGHTFIELD_PCA_2)),
        ], dim=1)
        
        # normalize according to pca stats to better cover the full 0-1 range for visualization
        t = t / tensor([0.25,0.11,0.02])
        brighten_factor = 5
        t = t*brighten_factor
        t = t.view(h,w,-1)
        t = t.clamp(0.,1.).permute(2,0,1)
        
        return show_image(TensorImage(t), title='Brightfield', ctx=ctx, **kwargs)
    
class FluorescenceTile(Tensor):
    @classmethod
    def create(cls, fn, stats:DataStats):
        im = TensorImage(np.load(fn).astype(np.float32))
        
        # normalize
        im = (im - tensor(stats.FLUORESCENCE_MEAN)) / tensor(stats.FLUORESCENCE_STD)
        
        return cls(im.permute(2,0,1))
    
    def to_numpy(self, stats:DataStats):
        """ Denormalize back to numpy array """
        img = self
        if not isinstance(img, Tensor):
            t = tensor(img)
            t = t.permute(2,0,1)
        else: 
            t = img
            
        t = t.permute(1,2,0)
        im = (t * tensor(stats.FLUORESCENCE_STD)) + tensor(stats.FLUORESCENCE_MEAN)
        np_im = im.cpu().numpy()
        np_im = np.round(np_im) #rounding to preserve the accuracy
        np_im = np.clip(np_im, 0, np.iinfo(np.uint16).max) #clipping to prevent modulo errors
        np_im = np_im.astype(np.uint16) #casting to uint16
        
        return np_im
    
    def show(self, ctx=None, **kwargs):
        img = self
        if not isinstance(img, Tensor):
            t = tensor(img)
            t = t.permute(2,0,1)
        else: 
            t = img
            
        t = t.permute(1,2,0)
        
        # cover the full 0-1 range for visualization
        t = (t + 2.) / 4.0
        t = t.clamp(0.,1.).permute(2,0,1)
        
        if 'stats' in kwargs: kwargs.pop('stats')
        
        return show_image(TensorImage(t), title='Fluorescence', ctx=ctx, **kwargs)