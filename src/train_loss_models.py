import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
import cv2
from tqdm.notebook import tqdm

from fastai.vision.all import *
from fastai.vision import models
import torch

from absl import app
from absl import flags

import sys
sys.path.append('./')
from src.types.tensor_tiles import FluorescenceTile
from src.stats_reader import DataStats
from src.transforms.augmentations import PairAugmentations
from src.types.fluorescence_tuple import FluorescenceTuple
from src.transforms.contrastive_transform import ContrastiveFluorescenceTransform
from src.models.loss_model import get_loss_model_vgg, siamese_splitter

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/train_config.json', 'Config file includes dataset specific parameters')
flags.DEFINE_integer('bs', 4, 'batch size')
flags.DEFINE_integer('ep_head', 5, 'Head-only training epochs')
flags.DEFINE_float('lr_head', 0.001, 'Head-only training learning rate')
flags.DEFINE_integer('ep_unfreezed', 25, 'unfreezed training epochs')
flags.DEFINE_float('lr_unfreezed', 0.0001, 'unfreezed training learning rate')

def main(unused_argv):
    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)

    required_keys = ['tile_data_dir', 'test_fold_index', 'stats_path', 'perceptual_loss_20_path',
    'perceptual_loss_40_path', 'perceptual_loss_60_path']
    for required_key in required_keys:
        assert required_key in config.keys(), f'config missing {required_key} key'

    stats_path = config['stats_path']
    test_fold = config['test_fold_index']
    
    for zoom, output_path in zip(
        [60,40,20],
        [config['perceptual_loss_60_path'], config['perceptual_loss_40_path'], config['perceptual_loss_20_path']]):

        STATS = DataStats(stats_path, zoom)
        path = config['tile_data_dir']
        df = pd.read_csv(os.path.join(path, f'train_{zoom}.csv'))

        train_fluorescence_paths = [os.path.join(path, fn) for fn, fold in zip(df.target_fn.values, df.fold.values) if fold != test_fold]
        valid_fluorescence_paths = [os.path.join(path, fn) for fn, fold in zip(df.target_fn.values, df.fold.values) if fold == test_fold]

        augmentations = PairAugmentations()

        train_tl= TfmdLists(range(len(train_fluorescence_paths)), ContrastiveFluorescenceTransform( 
            train_fluorescence_paths,
            stats=STATS,
            augment_func=augmentations,
            is_valid=False
        ))

        valid_tl= TfmdLists(range(len(valid_fluorescence_paths)), ContrastiveFluorescenceTransform( 
            valid_fluorescence_paths,
            stats=STATS,
            augment_func=augmentations,
            is_valid=True
        ))

        dls = DataLoaders.from_dsets(train_tl, valid_tl, bs=FLAGS.bs)
        dls = dls.cuda()

        model = get_loss_model_vgg()

        learn = Learner(dls,
                        model, 
                        loss_func=CrossEntropyLossFlat(), 
                        splitter=siamese_splitter, 
                        metrics=accuracy)
        
        # train only the head
        learn.freeze()
        learn.fit_one_cycle(FLAGS.ep_head, FLAGS.lr_head)

        # callbacks
        cbs=[
            EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=8),
            SaveModelCallback(fname='loss_model') # saves the best according to valid_loss
        ]
        
        # train unfreezed model with variable lrs
        learn.unfreeze()
        learn.fit_one_cycle(
            FLAGS.ep_unfreezed, 
            slice(FLAGS.lr_unfreezed * 0.1, FLAGS.lr_unfreezed),
            cbs=cbs
        )

        torch.save(model.state_dict(), output_path)
        print(f'model saved to {output_path}')

if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)