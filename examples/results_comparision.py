import os
import sys
base_path = '/home/cc/projects/nowcasting'#"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)

import torch
import h5py
import datetime
import numpy as np  
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns



from servir.datasets.dataLoader_ghana_imerg_h5 import load_wa_imerg_data_from_h5

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)


method_name = 'ConvLSTM'
add_naive = True


st = '2020-08-25' 
ed = '2020-09-01'

# prediction file name
# pred_fnames = ['imerglog_gtIRthr_SepTrue_L2ch_predictions.h5',\
#                 'imerglog_gtIRthr_SepTrue_L1ch_predictions.h5', \
#                 'imerglog_gtIRthr_SepFalse_L2ch_predictions.h5', \
#                 'imerglog_gtIRthr_SepFalse_L1ch_predictions.h5']

# pred_labels = ['SepTrue_L2', 'SepTrue_L1', 'SepFalse_L2', 'SepFalse_L1']

pred_fnames = ['wa_imerg/imerg01r_predictions.h5',\
            'wa_imerg_IR/imerg01r_gtIR01r_SepTrue_L1ch_predictions.h5',\
            'wa_imerg_IR/imerg01r_gtIR01r_SepTrue_L2ch_predictions.h5', \
            'wa_imerg_IR/imerg01r_gtIR01r_SepFalse_L2ch_predictions.h5']

pred_labels = [ 'imerg', 'withIR_TL1', 'withIR_TL2', 'withIR_FL2']

if add_naive:
    pred_fnames = ['wa_imerg/imerg_naive.h5'] + pred_fnames
    pred_labels = ['naive'] + pred_labels


in_seq_length = 12
out_seq_length = 12 

base_results_path = os.path.join(base_path, 'results')
# true
with h5py.File(os.path.join(base_results_path, 'wa_imerg/imerg_true.h5'), 'r') as hf:
    true = hf['precipitations'][:]

results_list = []
for pred_fname, pred_label in zip(pred_fnames, pred_labels):

    with h5py.File(os.path.join(base_results_path, pred_fname), 'r') as hf:
        pred_imgs = hf['precipitations'][:]


    # calculate mse
    mse = np.mean((pred_imgs - true)**2, axis=(2,3))

    c_list = []
    for k in range(out_seq_length):
        mse_k = mse[:, k]
        temp_df = pd.DataFrame(data = mse_k, columns = ['value'])
        temp_df['hours_ahead'] = (k+1)*0.5

        c_list.append(temp_df)

    c_df = pd.concat(c_list, ignore_index=True)
    c_df['label'] = pred_label

    results_list.append(c_df)   

results_df = pd.concat(results_list, ignore_index=True)
results_df['metric'] = 'mse'

# box plot
sns.boxplot(data=results_df, x='hours_ahead', y='value', hue='label')


# ['label', 'hours_ahead', 'metric', 'value'] 

#     fss = np.mean((np.sum(pred_imgs, axis=0) - np.sum(true, axis=0))**2) / (np.mean(np.sum(pred_imgs, axis=0)**2) + np.mean(np.sum(true, axis=0)**2))

#     results = results.append({'label': pred_label, 'hours_ahead': out_seq_length, 'mse': mse, 'fss': fss}, ignore_index=True)





