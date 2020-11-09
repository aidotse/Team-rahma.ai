from shutil import copyfile
import pandas as pd
import os
from glob import glob
import json

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/train_config.json', 'Config file with data paths')
flags.DEFINE_string('input_images_dir', './input/images_for_preview/', 'Images directory that has folders "20x images" e.g.')
flags.DEFINE_string('output_dir', './tmp/test_slides/', 'test slides are copied here')


def main(unused_argv):
    
    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)

    test_fold = int(config['test_fold_index'])
    IMG_DIR = FLAGS.input_images_dir
    OUT_DIR = FLAGS.output_dir
    
    df = pd.read_csv(config[''])
        
if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)