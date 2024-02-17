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
dataset_name = 'wa_imerg' #'wa_imerg_IR'

# prediction file name
base_fname = 'imerg_only_mse_relu'
pred_fname = f'{base_fname}_predictions.h5'

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')


# true imerg data path
dataPath1 = os.path.join(base_path, 'data', 'wa_imerg')
data1_fname = os.path.join(dataPath1, 'wa_imerg.h5')

# Load the ground truth
with h5py.File(data1_fname, 'r') as hf:
    imgs = hf['precipitations'][:]
    img_dts = hf['timestamps'][:]
    img_dts = [x.decode('utf-8') for x in img_dts]

img_datetimes = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in img_dts])

IR_norm = False
if dataset_name == 'wa_imerg_IR':
    # true ir data path
    dataPath2 = os.path.join(base_path, 'data', 'wa_IR')
    data2_fname = os.path.join(dataPath2, 'wa_IR.h5')

    if base_path == '/home/cc/projects/nowcasting':
        data2_fname = os.path.join(dataPath2, 'wa_IR_08.h5')

    with h5py.File(data2_fname, 'r') as hf:
        IRs = hf['IRs'][:]
        IR_times = hf['timestamps'][:]
        IR_times = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in IR_times]

    if IR_norm:

        st_dt = datetime.datetime.strptime('2020-08-25', '%Y-%m-%d')
        end_dt = datetime.datetime.strptime('2020-09-01', '%Y-%m-%d')

        ind = (np.array(IR_times)>=st_dt) & (np.array(IR_times)<end_dt)

        IRs = IRs[ind]
        IR_times = np.array(IR_times)[ind]
        IR_times = list(IR_times)

        IR_max = 336.0
        IR_min = 108.0
        IRs_norm = 1 -  (IRs - IR_min) / (IR_max - IR_min)
        IRs = IRs_norm


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


# Load the predictions
with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
    pred_imgs = hf['precipitations'][:]
    output_dts = hf['timestamps'][:]
    output_dts = [x.decode('utf-8').split(',') for x in output_dts]

# specify the gif output path
results_path = os.path.join(base_results_path, base_fname)
if not os.path.exists(results_path):
    os.mkdir(results_path)  

losses = []
# For each senario, match the input, true, and pred images.
for i, output_dt_i in enumerate(output_dts):
    # path to save the current sample images
    i_path = os.path.join(results_path, f'{i}')
    if not os.path.exists(i_path):
        os.mkdir(i_path)
        os.mkdir(os.path.join(i_path, 'true'))
        os.mkdir(os.path.join(i_path, 'pred'))
        # os.mkdir(os.path.join(i_path, 'input'))
        if dataset_name == 'wa_imerg_IR':
            os.mkdir(os.path.join(i_path, 'IR'))

    
    # locate the index of output index for sample i
    output_ind_i = np.array([img_dts.index(x) for x in output_dt_i])
    input_ind_i = output_ind_i - out_seq_length  

    in_dt_i = [img_datetimes[x] for x in input_ind_i]
    out_dt_i = [img_datetimes[x] for x in output_ind_i]

    if dataset_name == 'wa_imerg_IR':
        # locate IR images for sample i
        output_ind_IR_i = [IR_times.index(x) for x in out_dt_i]
        output_IRs_i = IRs[output_ind_IR_i, :, :]
        for k in range(output_IRs_i.shape[0]):

            tstr = IR_times[output_ind_IR_i[k]].strftime('%Y%m%d%H%M')
            plt.imshow(output_IRs_i[k], cmap='gray')
            plt.savefig(os.path.join(i_path, 'IR', f'{tstr}.png'))


    # # locate the input images for sample i
    # input_imgs_i = imgs[input_ind_i, :, :]
    # create_precipitation_gif(input_imgs_i, in_dt_i, timestep_min, wa_imerg_metadata, 
    #                         os.path.join(i_path, 'input'), title=f'{i} - input', gif_dur = 1000)

    # locate the ground truth images for sample i
    true_imgs_i = imgs[output_ind_i, :, :]
    create_precipitation_plots(true_imgs_i, out_dt_i, timestep_min, wa_imerg_metadata,\
                            os.path.join(i_path, 'true'), title=f'{i} - true')
    
    # locate the predicted images for sample i
    pred_imgs_i = pred_imgs[i, :, :, :]
    create_precipitation_plots(pred_imgs_i, out_dt_i, timestep_min, wa_imerg_metadata, \
                            os.path.join(i_path, 'pred'), f'{i} - pred')

    mse_i = np.mean((true_imgs_i - pred_imgs_i)**2) 
    losses.append(mse_i)
    print(mse_i)


print(np.mean(losses))

    

# plt.boxplot(losses)
# print('stop for debug')