import os
import json

class DataStats():
    """
    Utility class for accessing data statistics.
    """
    def __init__(self, fn='./configs/data_statistics.json', zoom=60, stats=None):
        """
        Arguments:
            fn     (str): configs json path, not used if json is provided
            zoom   (int): zoom integer, required only if fn is provided
            stats (dict): stats dict for a specific zoom
        """
        if stats is not None:
            self._load_stats(stats)
        elif os.path.isfile(fn):
            with open(fn) as json_file:
                stats = json.load(json_file)
                stats = stats[f'zoom_{zoom}']
            self._load_stats(stats)
        elif os.path.isfile(os.path.join('..',fn)):
            with open(os.path.join('..',fn)) as json_file:
                stats = json.load(json_file)
                stats = stats[f'zoom_{zoom}']
            self._load_stats(stats)
        else:
            raise Exception('file not found', 'could not find data_statistics.json from ./configs/ or ../configs/')
            
    # load statistics from json
    def _load_stats(self, stats):
        self.stats = stats
        
        self.BRIGHTFIELD_MEAN = stats['input_mean']
        self.BRIGHTFIELD_STD = stats['input_std']

        self.BRIGHTFIELD_PCA_0 = stats['input_pca_comp_0']
        self.BRIGHTFIELD_PCA_1 = stats['input_pca_comp_1']
        self.BRIGHTFIELD_PCA_2 = stats['input_pca_comp_2']

        self.FLUORESCENCE_MEAN = stats['target_mean']
        self.FLUORESCENCE_STD = stats['target_std']
        
    def to_dict(self):
        return self.stats

