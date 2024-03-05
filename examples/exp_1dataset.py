import os
import sys
base_path ="/home1/zhang2012/nowcasting/"# '/home/cc/projects/nowcasting'#
sys.path.append(base_path)

import torch
import h5py 
import logging
import numpy as np


from servir.core.distribution import get_dist_info
from servir.core.trainer import train
# from servir.datasets.dataLoader_wa_imerg import waImergDataset, waImergDataset_withMeta
from servir.datasets.dataLoader_wa_IR import IRDataset, IRDataset_withMeta
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_config_info, logging_method_info
from servir.utils.main_utils import print_log

from servir.methods.ConvLSTM import ConvLSTM


#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_IR'
data_fname = 'wa_IR.h5'

dataLoaderFunc = IRDataset
dataLoaderFuncMeta = IRDataset_withMeta


normalize_method = '01range'

train_st = '2020-06-01' 
train_ed = '2020-08-18' 
val_st = '2020-08-18'
val_ed = '2020-08-25'
test_st = '2020-08-25' 
test_ed = '2020-09-01'

# file names
base_fname = f'{dataset_name}_{normalize_method[:3]}'
model_para_fname = f'{base_fname}_params.pth'
checkpoint_fname = f'{base_fname}_checkpoint.pth'
pred_fname = f'{base_fname}_predictions.h5'

print(f'base file - {base_fname}')

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)

print_log(f'config file at {config_path} logged')

# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    data_fname = 'wa_IR_08.h5'

    model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
    checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 
    pred_fname = pred_fname.split('.')[0] + '_local.h5'

    train_st = '2020-08-25'
    train_ed = '2020-08-28' 
    val_st = '2020-08-28'
    val_ed = '2020-08-30' 
    test_st = '2020-08-30'
    test_ed = '2020-09-01'

    config['batch_size'] = 2
    config['val_batch_size'] = 2
    config['num_hidden'] = '32, 32' 
    config['max_epoch'] = 10
    config['early_stop_epoch'] = 2 # test run on local machine

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print_log(f'results path : {base_results_path}')

#================================================#

# log config
logging_config_info(config)
print('configuration file logged')

##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', dataset_name)
fname = os.path.join(dataPath, data_fname)


trainSet = dataLoaderFunc(fname, start_date = train_st, end_date = train_ed,\
                        in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'],\
                        normalize_method=normalize_method)


valSet = dataLoaderFunc(fname, start_date = val_st, end_date = val_ed,\
                        in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'], \
                        normalize_method=normalize_method)

print('Dataset created.')
print_log(f'training_len = {len(trainSet)}')
print_log(f'val_len = {len(valSet)}')


dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=True, pin_memory=True) 

# update config
config['steps_per_epoch'] = len(dataloader_train)
##==================Setup Method=====================##

if (config['use_gpu']) and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
    # gpu = torch.cuda.get_device_properties(device)
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)

# # log method info
logging_method_info(config, method, device)
print('method setup')
#==============Distribution=========================##

# setup distribution
config['rank'], config['world_size'] = get_dist_info()

##==================Training=========================##
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, model_para_fname)
print_log(f'model parameters saved at {para_dict_fpath}')

checkpoint_fpath = os.path.join(base_results_path, checkpoint_fname)
logging.info(f'model training checkpoint saved at {checkpoint_fpath}')

train(dataloader_train, dataloader_val, method, config, para_dict_fpath, checkpoint_fpath)    

print(f"TRAINING DONE! Best model parameters saved at {para_dict_fpath}")

#======================================
testSet = dataLoaderFuncMeta(fname, start_date = test_st, end_date = test_ed,\
                                in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'], \
                                normalize_method=normalize_method)


dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# Loads best model’s parameter dictionary 
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))


test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname.split('.')[0]+'_raw.h5'),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

logging.info(f"PREDICTION DONE! Raw Prediction file saved at {os.path.join(base_results_path, pred_fname.split('.')[0]+'_raw.h5')}")


with h5py.File(fname, 'r') as hf:
    mean = hf['mean'][()]   
    std = hf['std'][()]
    max_value = hf['max'][()]
    min_value = hf['min'][()]
    

threshold=0.1

# imerg convert to mm/hr (need to be updated)
if normalize_method == 'gaussian':
    test_pred = test_pred * std + mean
elif normalize_method == '01range':
    test_pred = test_pred * (max_value - min_value) + min_value
elif normalize_method == 'log_norm':
    test_pred = np.where(test_pred < np.log10(threshold), 0.0, np.power(10, test_pred))

# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

logging.info(f"PREDICTION DONE! Prediction file saved at {os.path.join(base_results_path, pred_fname)}")
            
    










