from fastai.vision.all import *
from src.types.tensor_tiles import BrightfieldTile, FluorescenceTile
from src.stats_reader import DataStats
from src.types.parent_slide import TileTypes

class PairTransform(Transform):
    def __init__(self,
                 brightfield_paths:list, 
                 fluorescence_paths:list,
                 stats:DataStats,
                 augment_func:callable=None,
                 augment_samples:list=None,
                ):
        """
        Apply transforms for brightfield and fluorescence images.
        
        Arguments:
            brightfield_paths   (list:str): brightfield numpy array file paths
            fluorescence_paths  (list:str): fluorescence image paths paths
            stats              (DataStats): statistics for data normalization
            augment_func        (callable): augmentation function that takes in dictionary with
                                            ImageTypes.brightfield and ImageTypes.fluorescence keys
            augment_samples    (list:bool): boolean list that tells which samples are augmented
        """
        self.brightfield_paths = brightfield_paths
        self.fluorescence_paths = fluorescence_paths
        self.augment_func = augment_func
        self.augment_samples = augment_samples
        self.stats = stats
        
        assert (len(self.brightfield_paths) == len(self.fluorescence_paths)) or for_inference, 'BF and FS path lists lengths are different'
        
        if augment_func is not None and augment_samples is not None:
            assert len(self.brightfield_paths) == len(augment_samples),  'BF, FS and augment_booleans lengths shpuld be equal'
        
        # Checks
        for fn in self.brightfield_paths + self.fluorescence_paths:
            assert os.path.isfile(fn), f'{fn} not found, please check your paths'
        
    def __len__():
        return len(self.brightfield_paths)
        
    def encodes(self, idx):
        bf_img = BrightfieldTile.create(self.brightfield_paths[idx], self.stats)
        fs_img = FluorescenceTile.create(self.fluorescence_paths[idx], self.stats)

        if self.augment_func is not None and self.augment_samples is not None and self.augment_samples[idx]:
            augmented = self.augment_func({TileTypes.brightfield: bf_img, TileTypes.fluorescence: fs_img})
            bf_img, fs_img = augmented[TileTypes.brightfield], augmented[TileTypes.fluorescence]
        
        return bf_img, fs_img

@typedispatch
def show_batch(x:BrightfieldTile, y:FluorescenceTile, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    n = min(x.shape[0], max_n)*4
    ctxs = get_grid(n, nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs):
        if i % 2 == 0:
            BrightfieldTile(x[i//2]).show(ctx=ctx, **kwargs)
        else:
            FluorescenceTile(y[i//2]).show(ctx=ctx, **kwargs)
            
@typedispatch
def show_results(x:BrightfieldTile, y:FluorescenceTile, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=3, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
        
    n = min(x.shape[0], max_n)*4
    ctxs = get_grid(n, nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs):
        if i % 3 == 0:
            BrightfieldTile(x[i//3]).show(ctx=ctx, **kwargs)
        elif i % 3 == 1:
            FluorescenceTile(y[i//3]).show(ctx=ctx, **kwargs)
        else:
            FluorescenceTile(outs[i//3][0]).show(ctx=ctx, **kwargs)