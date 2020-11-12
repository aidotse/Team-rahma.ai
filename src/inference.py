from tqdm.auto import tqdm
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
import json

from absl import app
from absl import flags

sys.path.append('./')
sys.path.append('../')
from src.helper_factories import get_inference_func
from src.types.fluorescence_slide import FluorescenceSlide

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './models/20201109-193651_resnet50', 'Directory containing subfolders for models and each subdir includes zoom_20, zoom_40, zoom_60 dirs')
flags.DEFINE_integer('zoom', 60, 'magnification integer. One of 20, 40, 60')
flags.DEFINE_list('predict_files', None, 'comma-separated list of tiff file paths to predict. Each complete slide has 7 brightfield tiffs.')
flags.DEFINE_string('predict_dir', './tmp/test_slides/input_60_256/', 'Directory containing tiff files to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.')
flags.DEFINE_string('output_dir', './tmp/test_slides/predicted_60_256/', 'Directory where predicted fluorescence tiffs are saved to')

def predict(zoom, model_dir, predict_files=None, predict_dir=None, output_dir=None, use_perceptual_loss_model=True, disable_tqdm=False):
    """
    Predict brightfield tiff file list or folder with a given model_dir
    and save fluorescence tiff predictions to output_dir. Zoom level must be provided
    
    Arguments:
        model_dir       (str): Directory containing subfolders for models and each subdir includes zoom_20, zoom_40, zoom_60 dirs
        predict_files  (list): List of tiff file paths to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.
        predict_dir     (str): Directory containing *C04.tif files to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.
        output_dir      (str): Directory where predicted fluorescence tiffs are saved to
        use_perceptual_loss_model (bool): Default=True. If False, select only-mse-trained model if available 
    """
    
    inference_func = get_inference_func(model_dir)
    fluorescence_slides_list = inference_func(
        zoom=zoom,
        predict_files=predict_files,
        predict_dir=predict_dir,
        output_dir=output_dir,
        use_perceptual_loss_model=use_perceptual_loss_model,
        disable_tqdm = disable_tqdm
    )
        
    for fluorescence_slides in fluorescence_slides_list:
        
        ensemble_fluorescence_slide = FluorescenceSlide.fromFluorescenceSlides(
                fluorescence_slides, 
                name=fluorescence_slides[0].name
            )
        ensemble_fluorescence_slide.write_to(output_dir)
    
        
def main(unused_argv):
    # if this file is imported and not ran from shell, stop here
    if FLAGS.model_dir is not None:
        predict(
            zoom=FLAGS.zoom,
            model_dir=FLAGS.model_dir,
            predict_files=FLAGS.predict_files, 
            predict_dir=FLAGS.predict_dir, 
            output_dir=FLAGS.output_dir
        )
    
if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)