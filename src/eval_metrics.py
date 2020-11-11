## From https://github.com/aidotse/adipocyte_cell_challenge/blob/main/evaluation_code/hackathon_evaluation_metrics.ipynb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys
import os
import glob
import re 

def get_featurewise_mean_absolute_error(targ_file, pred_file):
    """The relative mean absolute error between two data sets. 

    Parameters
    ----------
    targ_file : str
        Path to csv file containing the CellProfiler results for ground truth images
        
    pred_file : str
        Path to csv file containing the CellProfiler results for generated  images

    Returns
    ----------
    mae_per_feature : array
        Mean absolute error (mae) for each feature in the dataset. Each feature-mae is normalized 
        with the corresponding feature median to account for different feature scales. 
    
    mae : float64
        Averaged mae_per_feature
    
    feature_names : object
        names of features in data set
        
    """
    
    # read the results into dataframes 
    df_targ = pd.read_csv(targ_file) 
    df_pred = pd.read_csv(pred_file)
    df_targ = df_targ.drop(['Metadata_Well', 'ImageNumber','Metadata_FoV'], axis=1) # drop metadata
    df_pred = df_pred.drop(['Metadata_Well', 'ImageNumber','Metadata_FoV'], axis=1) # drop metadata
    n_features = len(df_pred.columns) 
    feature_names = df_targ.keys()
    
    # feature normalization 
    median_targ = df_targ.median()
    df_targ = df_targ/median_targ
    df_pred = df_pred/median_targ

    # mean absolute error for each normalized feature 
    mae_per_feature = mean_absolute_error(df_pred, df_targ, multioutput='raw_values')
    
    # Take average of the mean absolute errors 
    mae = np.average(mae_per_feature)
    return mae, mae_per_feature, feature_names


def convert_images_to_array(image_dir):
    """Convert images in directory to numpy arrays 

    Parameters
    ----------
    image_dir : str
        Path to image directory. The directory must contain n image triplets 
        with the following naming convention
        AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C01.tif
        AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C02.tif
        AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C03.tif
        representing the three target channels for each well and field of view.

        
    Returns
    ----------
    y_c01 : array
        Array containing n C01 images. Shape: (n, image width, image height) 
    
    y_c02 : array
        Array containing n C02 images. Shape: (n, image width, image height) 
    
    y_c03 : array
        Array containing n C03 images. Shape: (n, image width, image height) 
        
    """
    # dataframe to store metadata for each file 
    df = pd.DataFrame(columns = ['path', 'pos', 'F', 'C', 'Z']) 
    df_row = pd.DataFrame(np.array([[0,0,0,0,0]]), columns = ['path', 'pos', 'F', 'C', 'Z']) 

    # get all files in image_dir
    for x in os.walk(image_dir):
        file_list = glob.glob(x[0] + '/AssayPlate*.tif')

    # get metadata from each file name and store in df
    for file in file_list:
        filename = os.path.split(file)[1]
        df_row['path'] = file
        df_row['pos'] = re.search(r'.\d\d_T',filename).group()[0:3]
        df_row['F'] = re.search(r'\dF\d\d\d',filename).group()[1:]
        df_row['C'] = re.search(r'\dC\d\d',filename).group()[1:]
        df_row['Z'] = re.search(r'\dZ\d\d',filename).group()[1:]
        df = df.append(df_row, sort=False)
    df = df.reset_index(drop=True)

    # group dataframe according to pos (well) and F (field of view)
    df_grouped = df.groupby(['pos','F'])

    # empty lists to store images for each flouresence channel
    y_c01 = []
    y_c02 = []
    y_c03 = []

    # for every group, load image from each channel  
    for state, frame in df_grouped:
        frame = frame.sort_values(by=['C','Z'])
        for row_index, row in frame.iterrows():
            if row['C'] == 'C01':
                im = plt.imread(row['path'])
                y_c01.append(im)
            elif row['C'] == 'C02':
                im = plt.imread(row['path'])
                y_c02.append(im)
            elif row['C'] == 'C03':
                im = plt.imread(row['path'])
                y_c03.append(im)

    # convert to numpy array
    y_c01 = np.array(y_c01)
    y_c02 = np.array(y_c02)
    y_c03 = np.array(y_c03)

    return y_c01, y_c02, y_c03

def get_pixelwise_mean_absolute_error(targ_dir, pred_dir):
    """Pixelwise realative mean absolute error between two image data sets
    
    Parameters
    ----------
    target_dir : str
        Path to image directory containing ground truth images.
    
    pred_dir : str
        Path to image directory containing generated images. T
        
    Both directories must contain n image triplets with the 
    following naming convention
    AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C01.tif
    AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C02.tif
    AssayPlate_Greiner_#655090_D02_T0001F007L01A01Z01C03.tif
    representing the three target channels for each well and field of view.

    Returns
    ----------
    mae_c01 : float64
        Pixel-to-pixel relative mean absolute error for channel C01 
    
    mae_c02 : float64
        Pixel-to-pixel relative mean absolute error for channel C02 
    
    mae_c03 : float64
        Pixel-to-pixel relative mean  absolute error for channel C03
    
    mae : float64
        Average of mae_c01, mae_c02 and mae_c03
    """
    # convert target images to array
    y_c01_targ, y_c02_targ, y_c03_targ = convert_images_to_array(targ_dir)
    # convert predicted images to array
    y_c01_pred, y_c02_pred, y_c03_pred = convert_images_to_array(pred_dir)
    
    # calculate mae between target and predicted images
    # mae is normalized to the target median 
    mae_c01 = mean_absolute_error(y_c01_targ.flatten(), y_c01_pred.flatten())/np.median(y_c01_targ)
    mae_c02 = mean_absolute_error(y_c02_targ.flatten(), y_c02_pred.flatten())/np.median(y_c02_targ)
    mae_c03 = mean_absolute_error(y_c03_targ.flatten(), y_c03_pred.flatten())/np.median(y_c03_targ)
    
    # take average of the mean absolute errors 
    mae = np.average([mae_c01, mae_c02, mae_c03])
    
    return mae, mae_c01, mae_c02, mae_c03