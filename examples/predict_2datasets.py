import os
import sys
base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting'#
sys.path.append(base_path)

import h5py 
import time
import torch
import logging

import numpy as np

from servir.core.distribution import get_dist_info
from servir.datasets.dataLoader_wa_imerg_IR import waImergIRDatasetTr_withMeta
from servir.utils.config_utils import load_config

from servir.methods.ConvLSTM import ConvLSTM

#================Specification=========================#
method_name = 'ConvLSTM'

dataset1_name = 'wa_imerg'
dataset2_name = 'wa_IR'

data1_fname = 'wa_imerg.h5'
data2_fname = 'wa_IR.h5'

# new data name
dataset_name = 'wa_imerg_IR'


st = '2020-08-25' 
ed = '2020-09-01'

channel_sep = True
loss_channels = 2
relu_last = False


imerg_normalize_method = '01range'
IR_normalize_method = '01range'

# file names
base_fname = f'imerg{imerg_normalize_method[:3]}_gtIR{IR_normalize_method[:3]}_Sep{channel_sep}_L{loss_channels}ch'

model_para_fname = f'{base_fname}_params.pth'
checkpoint_fname = f'{base_fname}_checkpoint.pth'
pred_fname = f'{base_fname}_predictions_IR.h5'
pred_fname_raw = f'{base_fname}_predictions_raw.h5'

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    print('results not found! Run experiments first!')

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)

print(f'config file at {config_path} logged')

config['channel_sep'] = channel_sep
config['loss_channels'] = loss_channels
config['relu_last'] = relu_last 



# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
    checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 
    pred_fname = pred_fname.split('.')[0] + '_local.h5'

    test_st = '2020-08-30'
    test_ed = '2020-09-01'

    data2_fname = 'wa_IR_08.h5'

    config['batch_size'] = 2
    config['val_batch_size'] = 2
    config['num_hidden'] = '32, 32' 
    config['max_epoch'] = 10
    config['early_stop_epoch'] = 2


##==================Data Loading=====================##
# where to load data
f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

testSet = waImergIRDatasetTr_withMeta(f1name, f2name, st, ed, \
                        in_seq_length=config['in_seq_length'],  out_seq_length=config['out_seq_length'], \
                        imerg_normalize_method=imerg_normalize_method, IR_normalize_method=IR_normalize_method)

dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)

# update config
config['steps_per_epoch'] = 10

# setup distribution
config['rank'], config['world_size'] = get_dist_info()
##==================Setup Method=====================##

if (config['use_gpu']) and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)

# Loads best model’s parameter dictionary 
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, model_para_fname)
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))


test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True, channel_sep=channel_sep)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname_raw),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

print(f"PREDICTION DONE! Raw Prediction file saved at {os.path.join(base_results_path, pred_fname_raw)}")


if config['channels'] > 1:
    test_pred = test_pred[:, :, 1:2, :, :]
    test_pred = np.squeeze(test_pred, axis=2)

with h5py.File(f1name, 'r') as hf:
    mean = hf['mean'][()]   
    std = hf['std'][()]
    max_value = hf['max'][()]
    min_value = hf['min'][()]
    
threshold=0.1

# imerg convert to mm/hr (need to be updated)
if imerg_normalize_method == 'gaussian':
    test_pred = test_pred * std + mean
elif imerg_normalize_method == '01range':
    test_pred = test_pred * (max_value - min_value) + min_value
elif imerg_normalize_method == 'log_norm':
    test_pred = np.where(test_pred < np.log10(threshold), 0.0, np.power(10, test_pred))


# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

print(f"PREDICTION DONE! Prediction file saved at {os.path.join(base_results_path, pred_fname)}")


            
