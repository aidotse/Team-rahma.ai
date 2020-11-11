from shutil import copyfile
import pandas as pd
import os
from glob import glob
import json
import sys
from tqdm import tqdm
import numpy as np

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/train_config.json', 'Config file with data paths')
flags.DEFINE_string('input_images_dir', '/data/', 'Images directory that has folders "20x images" e.g.')
flags.DEFINE_string('output_dir', './tmp/test_slides/', 'test slides are copied here')

def main(unused_argv):
    
    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)

    test_fold = int(config['test_fold_index'])
    tile_sz = int(config['tile_sz'])
    IMG_DIR = FLAGS.input_images_dir
    OUT_DIR = FLAGS.output_dir
    
    IMG_20_DIR = os.path.join(IMG_DIR, '20x_images')
    IMG_40_DIR = os.path.join(IMG_DIR, '40x_images')
    IMG_60_DIR = os.path.join(IMG_DIR, '60x_images')
    
    input_files_20 = glob(IMG_20_DIR + '/*C04.tif')
    input_files_40 = glob(IMG_40_DIR + '/*C04.tif')
    input_files_60 = glob(IMG_60_DIR + '/*C04.tif')

    target_files_20 = glob(IMG_20_DIR + '/*C01.tif') + glob(IMG_20_DIR + '/*C02.tif') + glob(IMG_20_DIR + '/*C03.tif')
    target_files_40 = glob(IMG_40_DIR + '/*C01.tif') + glob(IMG_40_DIR + '/*C02.tif') + glob(IMG_40_DIR + '/*C03.tif')
    target_files_60 = glob(IMG_60_DIR + '/*C01.tif') + glob(IMG_60_DIR + '/*C02.tif') + glob(IMG_60_DIR + '/*C03.tif')
    
    for (input_files, target_files, size_text) in tqdm(zip(
        [input_files_20, input_files_40, input_files_60],
        [target_files_20, target_files_40, target_files_60],
        ['20','40','60'])
        ):
        
        df = pd.read_csv(os.path.join(config['tile_data_dir'], f'train_{size_text}.csv'))
        df = df[df.fold == test_fold]
        test_slide_names = list(df.slide_name.values)
        

        input_dir = os.path.join(OUT_DIR, f'{size_text}x_input')
        target_dir = os.path.join(OUT_DIR, f'{size_text}x_target')

        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)
            
        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
            
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
            
        unique_slide_names = np.unique(np.array([(os.path.basename(fn).split('.')[0][:-9]) for fn in input_files]))
        unique_slide_names = [slide_name for slide_name in unique_slide_names if slide_name in test_slide_names]
        
        for unique_slide in unique_slide_names:
            slide_input_files = [fn for fn in input_files if unique_slide in fn]
            slide_target_files = [fn for fn in target_files if unique_slide in fn]
            
            # copy test fold input files
            for slide_input in slide_input_files:
                dst = os.path.join(input_dir, os.path.basename(slide_input))
                print(f'Copying {slide_input} to {dst}')
                copyfile(slide_input, dst)
            
            # copy target files
            for slide_target in slide_target_files:
                dst = os.path.join(target_dir, os.path.basename(slide_target))
                print(f'Copying {slide_target} to {dst}')
                copyfile(slide_target, dst)
                
    print('Done')
        
if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)