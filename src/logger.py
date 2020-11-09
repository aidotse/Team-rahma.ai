import json
import numpy as np
from datetime import datetime
import os

class Logger:
    """
    Helper class for creating and saving records to output directories.
    """

    def __init__(self, model_name='', zoom=None ,root="./models/", save_date=True):
        """
        Create a logger instance with output folder.
        
        Arguments:
            model_name (str): identifier for the model
            zoom       (int): if not None, logdir is a zoom sub-folder under date-model_name folder
            root       (str): Root directory where logdir is created. default='./models/'
            save_date (bool): save date part in logdir
        """
        self.logdir = self._create_output_folder(model_name=model_name, zoom=zoom ,root=root, save_date=save_date)
        self.log = None

    def _create_output_folder(self, model_name='', zoom=None ,root="./models/", save_date=True):
        
        def exists_or_mkdir(dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        
        dir_name = datetime.now().strftime("%Y%m%d-%H%M%S") if save_date else ''
        dir_name += f'_{model_name}'
        logdir = os.path.join(root, dir_name)
        exists_or_mkdir(root)
        exists_or_mkdir(logdir)
        
        # create zoom subdir
        if zoom is not None:
            logdir = os.path.join(logdir, f'zoom_{zoom}')
            exists_or_mkdir(logdir)
        
        return logdir

    def log_json(self, dictionary, fn='model_config.json'):
        """
        Log a dictionary as json.
        
        Arguments:
            dictionary (dict): dict to save (must be json serializable)
            fn           (fn): output json file, default = 'model_config.json'
        """
        with open(os.path.join(self.logdir,fn), 'w') as text_file:
            text_file.write(json.dumps(dictionary, indent=4))

    def append_log(self, text, fn='Log.txt', silent=True):
        """
        Log text. Appends if file exists. 
        
        Arguments:
            text        (str): dict to save (must be json serializable)
            fn           (fn): output text file, default = 'Log.txt'
            silent     (bool): if true, text is printed to console also
        """
        with open(os.path.join(self.logdir, fn), 'a+') as text_file:
            text_file.write(text + '\n')
        if not silent:
            print(text)