from tqdm import tqdm
import numpy as np
import os
from glob import glob
import cv2
import sys
from fastai.vision.all import *
from fastai.vision import models
import torch
import matplotlib.pyplot as plt
import pandas as df

from absl import app
from absl import flags

sys.path.append('./')
sys.path.append('../')
from src.logger import Logger
from src.types.tensor_tiles import BrightfieldTile, FluorescenceTile
from src.stats_reader import DataStats
from src.transforms.augmentations import PairAugmentations
from src.transforms.pair_transform import PairTransform
from src.helper_factories import get_learner
from src.losses import *

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/train_config.json', 'Config file includes dataset specific parameters')
flags.DEFINE_integer('bs', 8, 'batch size')
flags.DEFINE_list('val_folds', None, 'comma-separated list of validation indices to run e.g."0,1,2", if None (default), all folds except test fold are used for val')
flags.DEFINE_integer('head_epochs', 1, 'number of epochs to train the head')
flags.DEFINE_float('head_lr', 0.001, 'head training learning rate')
flags.DEFINE_integer('unfreezed_mse_epochs', 15, 'number of epochs to train the full model with mse loss')
flags.DEFINE_float('unfreezed_mse_lr', 0.0001, 'unfreezed training learning rate in MSE loss stage')
flags.DEFINE_integer('unfreezed_combination_loss_epochs', 10, 'number of epochs to train the full model with perceptual loss')
flags.DEFINE_float('unfreezed_combination_loss_lr', 0.0001, 'unfreezed training learning rate in perceptual loss stage')
flags.DEFINE_string('base_arch', 'resnet50', 'model base arch name, one of [resnet50, resnest50]')

def plot_eyeball_batch(ys, preds, STATS, save_fn, scale=4):
    """
    Plot and save eyeball batch visualization.
    Plots GT, Pred, and MSE of batch images.
    
    Arguments:
        ys       (tensor): batch of ground truth FluorescenceTiles
        preds    (tensor): predictions from ys
        STATS (DataStats): DataStats instance that contains FluorescenceTile statistics
        save_fn    (path): png save path e.g. 'vis.png'
        scale       (int): plotting scale, bigger value saves larger plots, default=4
        
    Returns:
        None
    """
    n_cols = 3
    brightness_f = 0.4 # reduce MAE brigthness by this factor 
    f, axs = plt.subplots(len(ys),n_cols, figsize=(scale*n_cols, scale*len(ys)))
    for i in range(len(ys)):
        ys[i].show(stats=STATS, ctx=axs[i,0])
        axs[i,0].set_title('GT')
        
        FluorescenceTile(preds[i].cpu()).show(stats=STATS, ctx=axs[i,1])
        axs[i,1].set_title('Prediction')
        
        a = tensor(ys[i]).cpu().numpy() * brightness_f
        b = tensor(preds[i]).cpu().numpy() * brightness_f
        mae = np.abs(a - b)
        TensorImage(tensor(mae)).show(ctx=axs[i,2])
        axs[i,2].set_title(f'MAE ({mae.mean():.3f})')
        
    if save_fn is not None:
        plt.savefig(save_fn, transparent=False)
        plt.close('all')

def train_single_fold(root_dir, val_fold, train_folds, zoom, config):
    """
    Trains single fold and logs models, parameters, and visualizations under root dir.
    
    Arguments:
        root_dir     (path): root directory where the subfolders with models are saved
        val_fold      (int): validation fold index
        train_folds ([int]): list of training fold indices
        zoom          (int): zoom level ineteger, one of [20,40,60]
        config       (dict): config json with data paths
        
    Returns:
        None
    """

    assert zoom in [20,40,60], f'unrecognized zoom {zoom}'
    assert val_fold not in train_folds, 'validation fold in train folds'
    
    LOGGER = Logger(model_name=FLAGS.base_arch + f'_fold_{val_fold}', zoom=zoom, root=root_dir, save_date=False)
    LOGGER.append_log(FLAGS.__str__())
    
    STATS = DataStats(config['stats_path'], zoom)

    # record training parameters to model config
    config['bs'] = FLAGS.bs
    config['base_arch'] = FLAGS.base_arch
    config['head_epochs'] = FLAGS.head_epochs
    config['head_lr'] = FLAGS.head_lr
    config['unfreezed_mse_epochs'] = FLAGS.unfreezed_mse_epochs
    config['unfreezed_mse_lr'] = FLAGS.unfreezed_mse_lr
    config['unfreezed_combination_loss_epochs'] = FLAGS.unfreezed_combination_loss_epochs
    config['unfreezed_combination_loss_lr'] = FLAGS.unfreezed_combination_loss_lr
    config['stats'] = STATS.to_dict() # record stats model config
    
    # save config to output dir
    LOGGER.log_json(config)
    
    csv_logger = CSVLogger(fname=os.path.join(LOGGER.logdir, 'history.csv'), append=True)

    tile_data_dir = config['tile_data_dir']
    df = pd.read_csv(os.path.join(tile_data_dir, f'train_{zoom}.csv'))
    
    # exclude test set
    test_fold = int(config['test_fold_index'])
    df = df[df.fold != test_fold]
    assert len(df[df.fold == test_fold]) == 0

    # paths and train/val split
    brightfield_paths = [os.path.join(tile_data_dir, fn) for fn in df.fn.values]
    fluorescence_paths = [os.path.join(tile_data_dir, fn) for fn in df.target_fn.values]
    is_train = [fold != 0 for fold in df.fold.values]
    df['is_val'] = ~np.array(is_train)

    # transforms
    augmentations = PairAugmentations()
    tfm = PairTransform(
        brightfield_paths, 
        fluorescence_paths, 
        stats=STATS,
        augment_func=augmentations,
        augment_samples=is_train
    )
    
    # splitter defines validation samples
    splitter = FuncSplitter(lambda o: df[df.fn == str(o).replace(tile_data_dir,'')].is_val.values[0])
    splits = splitter(brightfield_paths)
    
    # dataloader
    tls = TfmdLists(range(len(brightfield_paths)), tfm, splits=splits)
    dls = tls.dataloaders(bs=FLAGS.bs)
    dls = dls.cuda()

    # plot samples from the same validation batch in different stages of training
    eyeball_xs, eyeball_ys = dls.one_batch()
    
    learn = get_learner(
        base_arch=FLAGS.base_arch,
        dls=dls,
        loss_method='mse'
    )
    
    # load perceptual loss for finetuning phase
    perceptual_loss_path = os.path.join(config[f'perceptual_loss_{zoom}_path'])
    perceptual_loss = VGGTrainedPerceptualLoss(pretrained_path=perceptual_loss_path)
    
    # training callbacks
    cbs=[
        EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3),
        SaveModelCallback(fname='unet_model'), # saves the best according to valid_loss
        csv_logger
    ]

    # train only the head part (imagenet trained encoder stays frozen)
    learn.freeze()
    learn.fit_one_cycle(FLAGS.head_epochs, FLAGS.head_lr, cbs=[csv_logger])

    # plot visualization
    preds = learn.model(eyeball_xs).detach()
    plot_eyeball_batch(eyeball_ys, preds, STATS, save_fn = os.path.join(LOGGER.logdir, '1_head_trained_vis.png'))
    
    # train unfreezed model with variable lrs
    learn.fit_flat_cos(
        FLAGS.unfreezed_mse_epochs,
        slice(FLAGS.unfreezed_mse_lr / 2.0, FLAGS.unfreezed_mse_lr),
        cbs=cbs
    )

    # plot visualization
    preds = learn.model(eyeball_xs).detach()
    plot_eyeball_batch(eyeball_ys, preds, STATS, save_fn = os.path.join(LOGGER.logdir, '2_unfreezed_trained_vis.png'))

    # save the mse trained model (only weights)
    torch.save(
        learn.model.state_dict(),
        os.path.join(LOGGER.logdir, 'model_mse_trained.pth')
        )
    
    learn.loss_func = perceptual_loss
    learn.fit_flat_cos(
        FLAGS.unfreezed_combination_loss_epochs,
        slice(FLAGS.unfreezed_combination_loss_lr / 5.0, FLAGS.unfreezed_combination_loss_lr),
        cbs=cbs
    )

    # plot visualization
    preds = learn.model(eyeball_xs).detach()
    plot_eyeball_batch(eyeball_ys, preds, STATS, save_fn = os.path.join(LOGGER.logdir, '3_perceptual_trained_vis.png'))

    # save the final model (only weights)
    torch.save(
        learn.model.state_dict(),
        os.path.join(LOGGER.logdir, 'model.pth')
        )


def main(unused_argv):
    """
    Trains all zoom level models and all cv-folds
    """
    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)

    # check that config has all required fields
    required_keys = ['tile_data_dir', 'tile_sz', 'num_folds', 
    'test_fold_index', 'stats_path', 'perceptual_loss_20_path',
    'perceptual_loss_40_path', 'perceptual_loss_60_path']
    for required_key in required_keys:
        assert required_key in config.keys(), f'config missing {required_key} key'

    # Spare one test fold and use others for CV-training
    test_fold = int(config['test_fold_index'])
    all_train_folds = list(range(int(config['num_folds'])))
    all_train_folds.pop(test_fold)
    
    # create root dir for this run
    LOGGER = Logger(model_name=FLAGS.base_arch, zoom=None)
    root_dir = LOGGER.logdir
    
    val_folds = [int(fold) for fold in FLAGS.val_folds] if FLAGS.val_folds is not None else all_train_folds
    
    for val_fold in val_folds:
        train_folds = all_train_folds.copy()
        train_folds.pop(val_fold)
        assert val_fold not in train_folds
        assert test_fold not in train_folds
        assert len(train_folds) == int(config['num_folds']) - 2

        for zoom in [60,40,20]:
            train_single_fold(root_dir=root_dir, val_fold=val_fold, train_folds=train_folds, zoom=zoom, config=config)

if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)