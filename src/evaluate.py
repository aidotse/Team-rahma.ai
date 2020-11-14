import os, sys
import shutil
import json
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
import subprocess
from absl import app
from absl import flags

sys.path.append('./')
sys.path.append('../')
from src.predict_utils import predict
from src.eval_metrics import *

FLAGS = flags.FLAGS
flags.DEFINE_string('model_dir', './models/20201109-193651_resnet50', 'Directory containing subfolders for models and each subdir includes zoom_20, zoom_40, zoom_60 dirs')
flags.DEFINE_string('output_dir', None, 'Output directory to save the eval results')

# constants
PIPELINE_DIR = "./cellprofiler_pipelines/"
SLIDESET_ROOT = './tmp/test_slides/'
ZOOM_LEVELS = ["20", "40", "60"]
USE_PERCEPTUAL_LOSS = True
TMP_FOLDER = "/tmp"

# resolve data
all_tif_paths = [str(fn) for fn in Path(f"{SLIDESET_ROOT}").rglob("*.tif")]
all_tif_paths = sorted(all_tif_paths)
dataset_df = pd.DataFrame({"tif_path": all_tif_paths})
dataset_df['slide_id'] = dataset_df.tif_path.apply(lambda x: Path(x).stem.replace("AssayPlate_Greiner_#655090_", ""))
dataset_df['slide_no'] = dataset_df.slide_id.apply(lambda x: x[-4] if x[-1]=="4" 
                                                   else x[-1])
dataset_df['slide_id'] = dataset_df.slide_id.apply(lambda x: x[:-4]+"X"+x[-3:] if x[-1]=="4" else 
                                                   x[:-7]+"X"+x[-6:-4] +"X"+x[-3:-1]+"X")
dataset_df['slide_id'] = dataset_df.slide_id.apply(lambda x: x[:-7]+"X"+x[-6:-1]+"X")
dataset_df['zoom'] = dataset_df.tif_path.apply(lambda x: x.split("/")[2][:2])
dataset_df['is_target'] = dataset_df.tif_path.apply(lambda x: "target" in x)
dataset_df = dataset_df.sort_values(["zoom", "slide_no", ])

def run_inference(model_dir:str):
    
    for zoom in tqdm(ZOOM_LEVELS, desc="[inference] zoom level"):
    
        zoom_df = dataset_df.copy()[(dataset_df.zoom==zoom)]
        slide_ids = zoom_df.slide_id.unique()
        
        # Generate paths to store the predictions
        # NOTICE! All existing data in the output folders will be deleted
        pred_output_dir = f"{TMP_FOLDER}/{zoom}x_preds/"
        target_output_dir = f"{TMP_FOLDER}/{zoom}x_targets/"
        if os.path.isdir(pred_output_dir):
            shutil.rmtree(pred_output_dir)
        if os.path.isdir(target_output_dir):
            shutil.rmtree(target_output_dir)

        Path(pred_output_dir).mkdir(parents=True, exist_ok=True)
        Path(target_output_dir).mkdir(parents=True, exist_ok=True)

        for slide_id in tqdm(slide_ids, desc="[inference] predicting slides"):

            input_files = zoom_df[(zoom_df.slide_id == slide_id) & 
                                    (~zoom_df.is_target )
                                   ].tif_path.to_list()

            target_files = zoom_df[(zoom_df.slide_id == slide_id) & 
                                   (zoom_df.is_target )
                                  ].tif_path.to_list()

            assert len(input_files)==7
            assert len(target_files)==3

            # Inference starts here
            predict(zoom=f"{zoom}", 
                    model_dir=f"{model_dir}",
                    predict_files=input_files,
                    output_dir=pred_output_dir,
                    use_perceptual_loss_model=USE_PERCEPTUAL_LOSS,
                    disable_prints=True
                   )
            for tgt_fn in target_files:
                shutil.copyfile(tgt_fn, f"{target_output_dir}/{Path(tgt_fn).name}")

def _run_shell_command(cmd:str):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    p.communicate() 
    
def run_cellprofiler_evaluation():
        
    tasks = []
    for zoom in ZOOM_LEVELS:

        pred_output_dir = f"{TMP_FOLDER}/{zoom}x_preds/"
        target_output_dir = f"{TMP_FOLDER}/{zoom}x_targets/"

        cp_cmd_preds = f"cellprofiler -c -r "\
                 f"-p {os.path.abspath(PIPELINE_DIR)}/Adipocyte_pipeline_{zoom}x.cppipe "\
                 f"-o {os.path.abspath(pred_output_dir)}/csv/ "\
                 f"-i {os.path.abspath(pred_output_dir)}"
        tasks.append(cp_cmd_preds)
        
        cp_cmd_targets = f"cellprofiler -c -r "\
                 f"-p {os.path.abspath(PIPELINE_DIR)}/Adipocyte_pipeline_{zoom}x.cppipe "\
                 f"-o {os.path.abspath(target_output_dir)}/csv/ "\
                 f"-i {os.path.abspath(target_output_dir)}"
        tasks.append(cp_cmd_targets)
        
    n_cores = min(mp.cpu_count(), len(tasks))
    with mp.Pool(processes=n_cores) as pool:
        for _ in tqdm(pool.imap_unordered(_run_shell_command, tasks), 
                      total=len(tasks),
                      desc=f"[cellprofiler] multiprocessing with {n_cores} cores"):
            continue

def compute_eval_score(output_dir:str):
    
    eval_data = []
    for zoom in tqdm(ZOOM_LEVELS, desc="[evaluation] zoom level"):

        pred_output_dir = f"{TMP_FOLDER}/{zoom}x_preds/"
        target_output_dir = f"{TMP_FOLDER}/{zoom}x_targets/"
        pred_file = f"{os.path.abspath(pred_output_dir)}/csv/Adipocytes_Image.csv"
        targ_file = f"{os.path.abspath(target_output_dir)}/csv/Adipocytes_Image.csv"

        mae_cp, mae_per_feature, feature_names = get_featurewise_mean_absolute_error(targ_file, pred_file)
        dummy, _, _ = convert_images_to_array(target_output_dir)
        mae_pix,_,_,_ = get_pixelwise_mean_absolute_error(target_output_dir, pred_output_dir)

        mae = np.average([mae_cp, mae_pix])
        print(f"Zoom level {zoom}:")
        print(f"Pixelwise MAE {mae_pix}")
        print(f"CP feature MAE {mae_cp}")
        print(f"Total MAE {mae}")
        print(f"Number of slides {len(dummy)}")
        print("")
        sys.stdout.flush()
        
        eval_data.append(
            {'zoom':zoom,
             'data':
             {
                 'mae_cp':mae_cp,
                 'mae_pix':mae_pix,
                 'mae_total':mae,
                 'n_slides':len(dummy)
             }})
    
    eval_data = {
        'model': FLAGS.model_dir,
        'eval_data': eval_data
    }
    mae_tot = np.divide(
        np.sum([e['data']['mae_total']*e['data']['n_slides'] for e in eval_data['eval_data']]),
        np.sum([e['data']['n_slides'] for e in eval_data['eval_data']])
    )
    eval_data['total_score'] = mae_tot
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(eval_data).to_json(os.path.join(output_dir, 'eval.json'))
    
    print("")
    print("~ "*30)
    print(f"Final evaluation score {mae_tot}")
    print("~ "*30)
    print("")
    print(eval_data)
    sys.stdout.flush()
    
def main(unused_argv):
    # if this file is imported and not ran from shell, stop here
    print("")
    print("")
    sys.stdout.flush()
    if FLAGS.model_dir is not None:
        run_inference(FLAGS.model_dir)
        run_cellprofiler_evaluation()
        if FLAGS.output_dir is not None:
            compute_eval_score(FLAGS.output_dir)
        else:
            compute_eval_score(FLAGS.model_dir)
    
if __name__ == '__main__':
    FLAGS(sys.argv)
    app.run(main)