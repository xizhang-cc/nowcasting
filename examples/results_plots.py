import os
import sys
import h5py
import datetime

import numpy as np  
from matplotlib import pyplot as plt

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)

from servir.visulizations.gif_creation import create_precipitation_plots, create_precipitation_gif


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

# prediction file name
base_fname = 'imerg01r'#'imerg01r_gtIR01r_SepTrue_L2ch'#'imerglog_gtIRthr_SepTrue_L2ch'
pred_fname = f'{base_fname}_predictions.h5'
 
# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
# Load the predictions
with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
    pred_imgs = hf['precipitations'][:]
    output_dts = hf['timestamps'][:]
    output_dts = [x.decode('utf-8').split(',') for x in output_dts]

pred_imgs[pred_imgs < 0] = 0

# load true images
imerg_true_path = os.path.join(base_path, 'results', 'wa_imerg')
with h5py.File(os.path.join(imerg_true_path, 'imerg_true.h5'), 'r') as hf:
    true = hf['precipitations'][:]

withIR = False
IR_norm = False


if withIR:
    # true ir data path
    dataPath2 = os.path.join(base_path, 'data', 'wa_IR')
    data2_fname = os.path.join(dataPath2, 'wa_IR.h5')

    if base_path == '/home/cc/projects/nowcasting':
        data2_fname = os.path.join(dataPath2, 'wa_IR_08.h5')

    with h5py.File(data2_fname, 'r') as hf:
        IRs = hf['IRs'][:]
        IR_times = hf['timestamps'][:]
        IR_times = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in IR_times]



in_seq_length = 12
out_seq_length = 12 


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




# specify the gif output path
results_path = os.path.join(base_results_path, base_fname)
if not os.path.exists(results_path):
    os.mkdir(results_path)  


# For each senario, match the input, true, and pred images.
for i, output_dt_i in enumerate(output_dts):

    output_dt_i = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in output_dt_i]

    # path to save the current sample images
    i_path = os.path.join(results_path, f'{i}')
    if not os.path.exists(i_path):
        os.mkdir(i_path)
        os.mkdir(os.path.join(i_path, 'true'))
        os.mkdir(os.path.join(i_path, 'pred'))
        # os.mkdir(os.path.join(i_path, 'input'))
        if withIR == True:
            os.mkdir(os.path.join(i_path, 'IR'))

    # if dataset_name == 'wa_imerg_IR':
    #     # locate IR images for sample i
    #     output_ind_IR_i = [IR_times.index(x) for x in out_dt_i]
    #     output_IRs_i = IRs[output_ind_IR_i, :, :]
    #     for k in range(output_IRs_i.shape[0]):

    #         tstr = IR_times[output_ind_IR_i[k]].strftime('%Y%m%d%H%M')
    #         plt.imshow(output_IRs_i[k], cmap='gray')
    #         plt.savefig(os.path.join(i_path, 'IR', f'{tstr}.png'))


    # # locate the input images for sample i
    # input_imgs_i = imgs[input_ind_i, :, :]
    # create_precipitation_gif(input_imgs_i, in_dt_i, timestep_min, wa_imerg_metadata, 
    #                         os.path.join(i_path, 'input'), title=f'{i} - input', gif_dur = 1000)

    # locate the ground truth images for sample i
    true_imgs_i = true[i]
    create_precipitation_plots(true_imgs_i, output_dt_i, timestep_min, wa_imerg_metadata,\
                            os.path.join(i_path, 'true'), title=f'{i} - true')
    
    # locate the predicted images for sample i
    pred_imgs_i = pred_imgs[i, :, :, :]
    create_precipitation_plots(pred_imgs_i, output_dt_i, timestep_min, wa_imerg_metadata, \
                            os.path.join(i_path, 'pred'), f'{i} - pred')





    

# plt.boxplot(losses)
# print('stop for debug')
