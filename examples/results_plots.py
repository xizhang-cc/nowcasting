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
dataset_name = 'ghana_imerg'

normalize_method = '01range'
in_seq_length = 12
out_seq_length = 12 
# prediction file name
base_fname = f'{dataset_name}_{normalize_method[:3]}'
pred_fname = f'{base_fname}_predictions.h5'
 
# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')

# Load the predictions
with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
    pred_samples = hf['precipitations'][:]
    output_dts = hf['timestamps'][:]
    output_dts_str = [x.decode('utf-8').split(',') for x in output_dts]
    # convert to list of datetime objects
    output_dts =[[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in output_dts_s] for output_dts_s in output_dts_str]

# load the ground truth observations
true_path = os.path.join(base_path, 'data', dataset_name)
with h5py.File(os.path.join(true_path, 'ghana_imerg_2011_2020_oct.h5'), 'r') as hf:
    true = hf['precipitations'][:]
    times = hf['timestamps'][:]
    times = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in times]


# wa_imerg_metadata = {'accutime': 30.0,
#     'cartesian_unit': 'degrees',
#     'institution': 'NOAA National Severe Storms Laboratory',
#     'projection': '+proj=longlat  +ellps=IAU76',
#     'threshold': 0.0125,
#     'timestamps': None,
#     'transform': None,
#     'unit': 'mm/h',
#     'x1': -21.4,
#     'x2': 30.4,
#     'xpixelsize': 0.04,
#     'y1': -2.9,
#     'y2': 33.1,
#     'yorigin': 'upper',
#     'ypixelsize': 0.04,
#     'zerovalue': 0}
    

timestep_min = 30.0


# specify the gif output path
results_path = os.path.join(base_results_path, base_fname)
if not os.path.exists(results_path):
    os.mkdir(results_path)  

# gif_path = os.path.join(results_path, f'{base_fname}_gifs')
# if not os.path.exists(gif_path):
#     os.mkdir(gif_path)
# # For each senario, match the input, true, and pred images.
# for i, output_dt_i in enumerate(output_dts):

#     # output_dt_i = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in output_dt_i]

#     # locate the ground truth images index for sample i
#     true_ind_i = [times.index(output_dt_i[k]) for k in range(len(output_dt_i))]



#     # locate the ground truth images for sample i
#     true_imgs_i = true[true_ind_i]
#     create_precipitation_gif(true_imgs_i, output_dt_i, timestep_min,\
#                             results_path, title=f'{i} - true')
    
#     # locate the predicted images for sample i
#     pred_imgs_i = pred_samples[i, :, :, :]
#     create_precipitation_gif(pred_imgs_i, output_dt_i, timestep_min, \
#                             results_path, f'{i} - pred')

# flooding event on 2020-10-10
flooding_event_path = os.path.join(results_path, 'Oct_flooding_event')
if not os.path.exists(flooding_event_path):
    os.mkdir(flooding_event_path)

# # get full day true precipitation images
# find = [datetime.datetime(2020, 10, 9, 0, 0, 0) + datetime.timedelta(minutes=30*k) for k in range(48)]
# true_fimgs_ind = [times.index(x) for x in find]
# true_fimgs = true[true_fimgs_ind]

# create_precipitation_gif(true_fimgs, find, timestep_min, \
#                     flooding_event_path, '2020-10-09-precipitations')


for n in range(48):
    nind = [(datetime.datetime(2020, 10, 9, 0, 0, 0) + datetime.timedelta(minutes=30*(n+k))) for k in range(out_seq_length)]

    true_ind_n = [times.index(x) for x in nind]
    true_imgs_n = true[true_ind_n]
    create_precipitation_gif(true_imgs_n, nind, timestep_min,\
                            flooding_event_path, title=f'{n} - true')
    

    # locate the predicted images for n sample
    pred_imgs_n = pred_samples[output_dts.index(nind)]
    create_precipitation_gif(pred_imgs_n, nind, timestep_min, \
                            flooding_event_path, f'{n} - pred')

    #     # locate the ground truth images for sample i
#     true_imgs_i = true[true_ind_i]

#     # locate the predicted images for sample i
#     pred_imgs_i = pred_samples[i, :, :, :]
#     create_precipitation_gif(pred_imgs_i, output_dt_i, timestep_min, \
#                             results_path, f'{i} - pred')





# =======To Get individual images for each sample=======
    # # path to save the current sample images
    # i_path = os.path.join(results_path, f'{i}')
    # if not os.path.exists(i_path):
    #     os.mkdir(i_path)
    #     os.mkdir(os.path.join(i_path, 'true'))
    #     os.mkdir(os.path.join(i_path, 'pred'))


    # # locate the input images for sample i
    # input_imgs_i = imgs[input_ind_i, :, :]
    # create_precipitation_gif(input_imgs_i, in_dt_i, timestep_min, wa_imerg_metadata, 
    #                         os.path.join(i_path, 'input'), title=f'{i} - input', gif_dur = 1000)


    # # locate the input images for sample i
    # input_imgs_i = imgs[input_ind_i, :, :]
    # create_precipitation_gif(input_imgs_i, in_dt_i, timestep_min, wa_imerg_metadata, 
    #                         os.path.join(i_path, 'input'), title=f'{i} - input', gif_dur = 1000)

    # # locate the ground truth images for sample i
    # true_imgs_i = true[i]
    # create_precipitation_plots(true_imgs_i, output_dt_i, timestep_min, wa_imerg_metadata,\
    #                         os.path.join(i_path, 'true'), title=f'{i} - true')
    
    # # locate the predicted images for sample i
    # pred_imgs_i = pred_imgs[i, :, :, :]
    # create_precipitation_plots(pred_imgs_i, output_dt_i, timestep_min, wa_imerg_metadata, \
    #                         os.path.join(i_path, 'pred'), f'{i} - pred')
    
# =======To Get individual images for each sample=======

# plt.boxplot(losses)
# print('stop for debug')

# # get the IR channel
# test_pred = pred_imgs[:, :, 1:2, :, :]
# test_pred = np.squeeze(test_pred, axis=2)

# maxv = 336.4427574092979
# minv = 108.1460311660434

# convert to original scale
# test_pred_ori = (1-test_pred)*(maxv-minv) + minv

# # load true images
# imerg_true_path = os.path.join(base_path, 'results', 'wa_imerg')
# with h5py.File(os.path.join(imerg_true_path, 'imerg_true.h5'), 'r') as hf:
#     true = hf['precipitations'][:]

# withIR = True
# IR_norm = False


# if withIR:
#     # true ir data path
#     dataPath2 = os.path.join(base_path, 'data', 'wa_IR')
#     data2_fname = os.path.join(dataPath2, 'wa_IR.h5')

#     if base_path == '/home/cc/projects/nowcasting':
#         data2_fname = os.path.join(dataPath2, 'wa_IR_08.h5')

#     with h5py.File(data2_fname, 'r') as hf:
#         IRs = hf['IRs'][:]
#         IR_times = hf['timestamps'][:]
#         IR_times = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in IR_times]

        # os.mkdir(os.path.join(i_path, 'input'))
        # if withIR == True:
        #     os.mkdir(os.path.join(i_path, 'IR'))

    # # if dataset_name == 'wa_imerg_IR':
    # # locate IR images for sample i
    # output_ind_IR_i = [IR_times.index(x) for x in output_dt_i]
    # output_IRs_i = IRs[output_ind_IR_i, :, :]
    # for k in range(output_IRs_i.shape[0]):

    #     tstr = IR_times[output_ind_IR_i[k]].strftime('%Y%m%d%H%M')
    #     plt.imshow(output_IRs_i[k], cmap='gray')
    #     plt.savefig(os.path.join(i_path, 'IR', f'{tstr}.png'))



    # for k in range(pred_imgs_i.shape[0]):

    #     plt.imshow(pred_imgs_i[k], cmap='gray')
    #     plt.savefig(os.path.join(i_path, 'pred', f'{output_dt_i[k]}.png'))



