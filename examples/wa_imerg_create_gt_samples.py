import os
import sys
import h5py
import datetime

import numpy as np  

base_path ='/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/" 
sys.path.append(base_path)
from servir.utils.config_utils import load_config
from servir.visulizations.gif_creation import create_precipitation_gif

method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

timestep_min = 30.0

# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 
config = load_config(config_path)

# data path
dataPath = os.path.join(base_path, 'data', dataset_name)
data_fname = os.path.join(dataPath, 'wa_imerg.h5')

# Load the ground truth
with h5py.File(data_fname, 'r') as hf:
    imgs = hf['precipitations'][:]
    img_dts = hf['timestamps'][:]
    img_dts = [x.decode('utf-8') for x in img_dts]

img_datetimes = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in img_dts])

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')

# Load the predictions
with h5py.File(os.path.join(base_results_path, 'imerg_only_mse_predictions.h5'), 'r') as hf:
    pred_imgs = hf['precipitations'][:]
    output_dts_s = hf['timestamps'][:]
    output_dts = [x.decode('utf-8').split(',') for x in output_dts_s]

gt_list = []
# For each senario, match the input, true, and pred images.
for i, output_dt_i in enumerate(output_dts):

    # locate the index of output index for sample i
    output_ind_i = np.array([img_dts.index(x) for x in output_dt_i])
    input_ind_i = output_ind_i - config['out_seq_length']  #[x - config['out_seq_length'] for x in output_ind_i]

    out_dt_i = [img_datetimes[x] for x in output_ind_i]

    # locate the ground truth images for sample i
    true_imgs_i = imgs[output_ind_i, :, :]
    # crop first and last column
    true_imgs_i =  true_imgs_i[:, :, 1:-1]

    gt_list.append(np.expand_dims(true_imgs_i, axis=0))

gt_array = np.concatenate(gt_list, axis=0)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, 'imerg_true.h5'),'w') as hf:
    hf.create_dataset('precipitations', data=gt_array)
    hf.create_dataset('timestamps', data=output_dts_s)











print('stop for debug')