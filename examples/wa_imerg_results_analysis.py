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

wa_imerg_metadata = {'accutime': 30.0,
    'cartesian_unit': 'degrees',
    'institution': 'NOAA National Severe Storms Laboratory',
    'projection': '+proj=longlat  +ellps=IAU76',
    'threshold': 0.0125,
    'timestamps': None,
    'transform': None,
    'unit': 'mm/h',
    'x1': -21.4,
    'x2': 30.4,
    'xpixelsize': 0.04,
    'y1': -2.9,
    'y2': 33.1,
    'yorigin': 'upper',
    'ypixelsize': 0.04,
    'zerovalue': 0}

timestep_min = 30.0


# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 
config = load_config(config_path)

# data path
dataPath = os.path.join(base_path, 'data', dataset_name)
data_fname = os.path.join(dataPath, 'imerg_2020_july.h5py')
# Load the ground truth
with h5py.File(data_fname, 'r') as hf:
    imgs = hf['precipitations'][:]
    img_dts = hf['timestamps'][:]
    img_dts = [x.decode('utf-8') for x in img_dts]

img_datetimes = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in img_dts])

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
# Load the predictions
with h5py.File(os.path.join(base_results_path, 'test_predictions.h5py'), 'r') as hf:
    pred_imgs = hf['precipitations'][:]
    output_dts = hf['timestamps'][:]
    output_dts = [x.decode('utf-8').split(',') for x in output_dts]

# convert to orignal sacles
pred_imgs = pred_imgs * 52


# specify the gif output path
gif_path = os.path.join(base_results_path, 'gifs')
if not os.path.exists(gif_path):
    os.mkdir(gif_path)  

# For each senario, match the input, true, and pred images.
for i, output_dt_i in enumerate(output_dts):

    # locate the index of output index for sample i
    output_ind_i = np.array([img_dts.index(x) for x in output_dt_i])
    input_ind_i = output_ind_i - config['out_seq_length']  #[x - config['out_seq_length'] for x in output_ind_i]

    in_dt_i = [img_datetimes[x] for x in input_ind_i]
    out_dt_i = [img_datetimes[x] for x in output_ind_i]


    # locate the input images for sample i
    input_imgs_i = imgs[input_ind_i, :, :]
    create_precipitation_gif(input_imgs_i, in_dt_i, \
                             timestep_min, wa_imerg_metadata, gif_path, f'{i} - input', gif_dur = 1000)

    # locate the ground truth images for sample i
    true_imgs_i = imgs[output_ind_i, :, :]
    create_precipitation_gif(true_imgs_i, out_dt_i, \
                             timestep_min, wa_imerg_metadata, gif_path, f'{i} - true', gif_dur = 1000)
    # locate the predicted images for sample i
    pred_imgs_i = pred_imgs[i, :, :, :]
    create_precipitation_gif(pred_imgs_i, out_dt_i, \
                             timestep_min, wa_imerg_metadata, gif_path, f'{i} - pred', gif_dur = 1000)






print('stop for debug')