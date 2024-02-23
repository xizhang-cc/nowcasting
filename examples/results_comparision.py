import os
import sys
import h5py
import datetime

import numpy as np  
import pandas as pd
from matplotlib import pyplot as plt

import torch

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)
from servir.datasets.dataLoader_wa_imerg_IR import waImergIRDatasetTr_withMeta


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg_IR'

add_naive = False
metrics = ['mse', 'fss']


st = '2020-08-25' 
ed = '2020-09-01'

# prediction file name
pred_fnames = ['imerglog_gtIRthr_SepTrue_L2ch_predictions.h5',\
                'imerglog_gtIRthr_SepTrue_L1ch_predictions.h5', \
                'imerglog_gtIRthr_SepFalse_L2ch_predictions.h5', \
                'imerglog_gtIRthr_SepFalse_L1ch_predictions.h5']

pred_labels = ['SepTrue_L2', 'SepTrue_L1', 'SepFalse_L2', 'SepFalse_L1']

if add_naive:
    pred_labels.append('naive') 


in_seq_length = 12
out_seq_length = 12 

#  np.arange(0.5, 6.5, 0.5).tolist()
results = pd.DataFrame(columns=['label', 'hours_ahead', 'mse', 'fss'])

# true imerg data path
dataPath1 = os.path.join(base_path, 'data', 'wa_imerg')
data1_fname = os.path.join(dataPath1, 'wa_imerg.h5')

# Load the ground truth
with h5py.File(data1_fname, 'r') as hf:
    imgs = hf['precipitations'][:]
    img_dts = hf['timestamps'][:]
    img_dts = [x.decode('utf-8') for x in img_dts]

img_datetimes = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in img_dts])

gt_list = []
# For each senario, match the input, true, and pred images.
for i, output_dt_i in enumerate(output_dts):

    # locate the index of output index for sample i
    output_ind_i = np.array([img_dts.index(x) for x in output_dt_i])
    # locate the ground truth images for sample i
    true_imgs_i = imgs[output_ind_i, :, :]

    gt_list.append(np.expand_dims(true_imgs_i, axis=0))

gt_array = np.concatenate(gt_list, axis=0)





# specify the results path
results_path = os.path.join(base_path, 'results')

if not os.path.exists(results_path):
    os.mkdir(results_path)  


for pred_fname in pred_fnames:
    # Results base path for logging, working dirs, etc. 
    base_results_path = os.path.join(base_path, f'results/{dataset_name}')
    # Load the predictions
    with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
        pred_imgs = hf['precipitations'][:]
        output_dts = hf['timestamps'][:]
        output_dts = [x.decode('utf-8').split(',') for x in output_dts]





