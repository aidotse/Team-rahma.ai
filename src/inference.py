import sys
import matplotlib.pyplot as plt
import json

from absl import app
from absl import flags

sys.path.append('./')
sys.path.append('../')
from src.predict_utils import predict

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './models/20201109-193651_resnet50', 'Directory containing subfolders for models and each subdir includes zoom_20, zoom_40, zoom_60 dirs')
flags.DEFINE_integer('zoom', 60, 'magnification integer. One of 20, 40, 60')
flags.DEFINE_list('predict_files', None, 'comma-separated list of tiff file paths to predict. Each complete slide has 7 brightfield tiffs.')
flags.DEFINE_string('predict_dir', './tmp/test_slides/input_60_256/', 'Directory containing tiff files to predict. Each complete slide has 7 brightfield tiffs. Either predict_files or predict_dir must be provided.')
flags.DEFINE_string('output_dir', './tmp/test_slides/predicted_60_256/', 'Directory where predicted fluorescence tiffs are saved to')    
        
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