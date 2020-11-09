from tqdm import tqdm
import numpy as np
import os
from glob import glob
import cv2
import sys
import json
from sklearn.decomposition import PCA

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_images_dir', './input/images_for_preview/', 'Images directory that has folders "20x images" e.g.')
flags.DEFINE_string('config', './configs/train_config.json', 'Config file that tells where to save stats')

def main(unused_argv):
    
    IMG_DIR = FLAGS.input_images_dir

    # read config
    with open(FLAGS.config) as json_file:
        config = json.load(json_file)
    output_file = config['stats_path']

    IMG_20_DIR = os.path.join(IMG_DIR, '20x_images')
    IMG_40_DIR = os.path.join(IMG_DIR, '40x_images')
    IMG_60_DIR = os.path.join(IMG_DIR, '60x_images')

    input_files_20 = glob(IMG_20_DIR + '/*C04.tif')
    input_files_40 = glob(IMG_40_DIR + '/*C04.tif')
    input_files_60 = glob(IMG_60_DIR + '/*C04.tif')

    target_files_20 = glob(IMG_20_DIR + '/*C01.tif') + glob(IMG_20_DIR + '/*C02.tif') + glob(IMG_20_DIR + '/*C03.tif')
    target_files_40 = glob(IMG_40_DIR + '/*C01.tif') + glob(IMG_40_DIR + '/*C02.tif') + glob(IMG_40_DIR + '/*C03.tif')
    target_files_60 = glob(IMG_60_DIR + '/*C01.tif') + glob(IMG_60_DIR + '/*C02.tif') + glob(IMG_60_DIR + '/*C03.tif')
    
    log = {}

    for (input_files, target_files, size_text) in zip(
        [input_files_20, input_files_40, input_files_60],
        [target_files_20, target_files_40, target_files_60],
        ['20','40','60']):

        zoom_log = {}

        # the folders have several slides and each slide includes channels as separate files
        # group all channel files for each slide

        # channel file name "AssayPlate_Greiner_#655090_D04_T0001F006L01A04Z01C04.tif"
        # contains name part "AssayPlate_Greiner_#655090_D04_T0001F006L01"
        # and channel part "A04Z01C04"
        unique_slide_names = np.unique(np.array([(os.path.basename(fn).split('.')[0][:-9]) for fn in input_files]))
        Xs = []
        ys = []
        for unique_slide in unique_slide_names:
            slide_input_files = [fn for fn in input_files if unique_slide in fn]
            slide_target_files = [fn for fn in target_files if unique_slide in fn]
            # Read raw input files and calculate stats
            X = np.stack([cv2.imread(fn, cv2.IMREAD_UNCHANGED) for fn in slide_input_files], axis=2).reshape(-1, len(slide_input_files))
            Xs.append(X.reshape((X.shape[0],-1)))

            y = np.stack([cv2.imread(fn, cv2.IMREAD_UNCHANGED) for fn in slide_target_files], axis=2).reshape(-1, len(slide_target_files))
            ys.append(y.reshape((y.shape[0],-1)))

        # take every 100th pixel to save time & memory
        X = np.concatenate(Xs, axis=0)[0::100]   
        y = np.concatenate(ys, axis=0)[0::100]
        
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        
        zoom_log['input_mean'] = list(X_mean)
        zoom_log['input_std'] = list(X_std)

        # Project to 3D PCA and calculate stats
        pca = PCA(n_components=3)
        pca.fit(X)
        pca_mean = pca.transform(X).mean(axis=0)
        pca_std = pca.transform(X).std(axis=0)
        pca_comps = pca.components_
        zoom_log['input_pca_mean'] = list(pca_mean)
        zoom_log['input_pca_std'] = list(pca_std)
        for i in range(len(pca_comps)):
            zoom_log[f'input_pca_comp_{i}'] = list(pca_comps[i])

        y_mean = y.mean(axis=0)
        y_std = y.std(axis=0)
        zoom_log['target_mean'] = list(y_mean)
        zoom_log['target_std'] = list(y_std)

        log[f'zoom_{size_text}'] = zoom_log
    
    with open(output_file, 'w') as f:
        json.dump(log, f, indent=4)

if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)