from tqdm import tqdm
import numpy as np
import os
from glob import glob
import cv2
import sys
import pandas as pd
import json
import sys
from sklearn.model_selection import GroupKFold

from absl import app
from absl import flags

sys.path.append('./')
from src.types.brightfield_slide import BrightFieldSlide
from src.types.fluorescence_slide import FluorescenceSlide

FLAGS = flags.FLAGS

flags.DEFINE_string('config', './configs/train_config.json', 'Config file that tells where to save and in with tile size')
flags.DEFINE_string('input_images_dir', '/data/', 'Images directory that has folders "20x images" e.g.')

def main(unused_argv):

    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)

    # check that config contains all required fields
    required_keys = ['tile_data_dir', 'tile_sz', 'tile_overlap', 'num_folds']
    for required_key in required_keys:
        assert required_key in config.keys(), f'config missing {required_key} key'

    IMG_DIR = FLAGS.input_images_dir
    OUT_DIR = config['tile_data_dir']
    tile_sz = config['tile_sz']
    tile_overlap = config['tile_overlap']
    gkf = GroupKFold(n_splits=config['num_folds'])

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
        input_dir = os.path.join(OUT_DIR, f'input_{size_text}_{tile_sz}')
        target_dir = os.path.join(OUT_DIR, f'target_{size_text}_{tile_sz}')
        
        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)

        if not os.path.isdir(input_dir):
            os.mkdir(input_dir)
            
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)
        
        # the folders have several slides and each slide includes channels as separate files
        # group all channel files for each slide

        # channel file name "AssayPlate_Greiner_#655090_D04_T0001F006L01A04Z01C04.tif"
        # contains name part "AssayPlate_Greiner_#655090_D04_T0001F006L01"
        # and channel part "A04Z01C04"

        df_rows = []
        unique_slide_names = np.unique(np.array([(os.path.basename(fn).split('.')[0][:-9]) for fn in input_files]))
        for unique_slide in unique_slide_names:
            slide_input_files = [fn for fn in input_files if unique_slide in fn]
            slide_target_files = [fn for fn in target_files if unique_slide in fn]

            # construct slides
            brightfield_slide = BrightFieldSlide.fromFiles(fns=slide_input_files, name=os.path.basename(unique_slide))
            fluorescence_slide = FluorescenceSlide.fromFiles(fns=slide_target_files, name=os.path.basename(unique_slide))

            # split to tiles
            brightfield_tiles = brightfield_slide.getTiles(tile_sz, tile_overlap)
            fluorescence_tiles = fluorescence_slide.getTiles(tile_sz, tile_overlap)

            assert len(brightfield_tiles) == len(fluorescence_tiles), f'there are {len(brightfield_tiles)} br tiles but {len(fluorescence_tiles)} fs tile'

            # save to files
            for br_tile, fs_tile in zip(brightfield_tiles, fluorescence_tiles):
                assert br_tile.start_x == fs_tile.start_x and br_tile.start_y == fs_tile.start_y, 'br and fs tile locations mismatch'
                br_fn = br_tile.write_file(dir=input_dir)
                target_fn = fs_tile.write_file(dir=target_dir)

                start_x = br_tile.start_x
                start_y = br_tile.start_y

                df_rows.append({
                    'fn'            : str(br_fn).replace(OUT_DIR,''),
                    'fold'          : -1,
                    'slide_name'    : unique_slide,
                    'start_x'       : int(start_x),
                    'start_y'       : int(start_y),
                    'tile_sz'       : int(tile_sz),
                    'slide_width'   : brightfield_slide.width,
                    'slide_height'  : brightfield_slide.height,
                    'target_fn'     : str(target_fn).replace(OUT_DIR,'')
                    })
        
        df = pd.DataFrame(df_rows)

        # split to folds, use slide names for groups but if there is only one slide, use start_y coords
        groups = df.slide_name.values if len(df.slide_name.unique()) > 1 else df.start_y.values
        folds = df.fold.values
        for fold, (_, test_index) in enumerate(gkf.split(X=df.index.values, groups=groups)):
            folds[test_index] = fold
        df.fold = folds
        assert -1 not in df.fold.values, 'error splitting folds'
        
        df.to_csv(os.path.join(OUT_DIR, f'train_{size_text}.csv'), index=False)

if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)