from fastai.vision.all import *
from src.stats_reader import DataStats
from src.types.parent_slide import TileTypes
from src.types.tensor_tiles import FluorescenceTile
from src.types.fluorescence_tuple import FluorescenceTuple

class ContrastiveFluorescenceTransform(Transform):
    def __init__(self,
                 fluorescence_paths:list,
                 stats:DataStats,
                 augment_func:callable,
                 is_valid=False
                ):
        """
        Apply transforms for brightfield and fluorescence images.
        
        Arguments:
            fluorescence_paths  (list:str): fluorescence image paths
            stats              (DataStats): statistics for data normalization
            augment_func        (callable): augmentation function that takes in dictionary with
                                            ImageTypes.brightfield and ImageTypes.fluorescence keys
            is_valid                (bool): is validation set (False by default)
        """
        self.fluorescence_paths = fluorescence_paths
        self.augment_func = augment_func
        self.stats = stats

         # Checks
        for fn in self.fluorescence_paths:
            assert os.path.isfile(fn), f'{fn} not found, please check your paths'

        self.is_valid = is_valid
        # draw validation set only once
        if is_valid: self.files2 = [self._draw(f) for f in fluorescence_paths]
       
    def __len__():
        return len(self.fluorescence_paths)
        
    def encodes(self, idx):
        file1 = self.fluorescence_paths[idx]
        (file2,same) = self.files2[idx] if self.is_valid else self._draw(file1)
        fs_img1 = FluorescenceTile.create(file1, self.stats)
        fs_img2 = FluorescenceTile.create(file2, self.stats)

        augmented = self.augment_func({TileTypes.fluorescence: fs_img1, TileTypes.fluorescence_contrastive: fs_img2})
        fs_img1, fs_img2 = augmented[TileTypes.fluorescence], augmented[TileTypes.fluorescence_contrastive]
        
        return FluorescenceTuple(fs_img1, fs_img2, int(same))

    def _draw(self, file1):
        same = random.random() < 0.5
        file2 = file1 if same else random.choice(self.fluorescence_paths)
        # change label if this is the same file after random pick
        same = (file1 == file2)
        return file2, same