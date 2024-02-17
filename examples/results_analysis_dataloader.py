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

add_naive = True
metrics = ['mse', 'fss']

st = '2020-08-25' 
ed = '2020-09-01'



# prediction file name
pred_fnames = ['imerg_r01_mse_predictions.h5', 'imerg_gtIR_r01_mse_predictions.h5']

pred_labels = ['imerg', 'imerg_relu', 'withIR_r01', 'withIR']

if add_naive:
    pred_labels.append('naive') 


in_seq_length = 12
out_seq_length = 12 

# true imerg data path
dataPath1 = os.path.join(base_path, 'data', 'wa_imerg')
data1_fname = os.path.join(dataPath1, 'wa_imerg.h5')

# Load the ground truth
with h5py.File(data1_fname, 'r') as hf:
    imgs = hf['precipitations'][:]
    img_dts = hf['timestamps'][:]
    img_dts = [x.decode('utf-8') for x in img_dts]

img_datetimes = np.array([datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in img_dts])

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

testSet = waImergIRDatasetTr_withMeta(data1_fname, data2_fname, st, ed, \
                        in_seq_length=in_seq_length,  out_seq_length=out_seq_length, \
                        imerg_normalize=False, IR_normalize=False)

dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# specify the gif output path
results_path = os.path.join(base_path, f'results/{dataset_name}')

if not os.path.exists(results_path):
    os.mkdir(results_path)  

results = pd.DataFrame(columns=['method', 'mse', 'fss'])

#



for pred_fname in pred_fnames:
    # Results base path for logging, working dirs, etc. 
    base_results_path = os.path.join(base_path, f'results/{dataset_name}')
    # Load the predictions
    with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
        pred_imgs = hf['precipitations'][:]
        output_dts = hf['timestamps'][:]
        output_dts = [x.decode('utf-8').split(',') for x in output_dts]



