import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")

import h5py
import torch
import pandas as pd

from servir.core.distribution import get_dist_info
from servir.core.trainer import train
from servir.datasets.dataLoader_wa_imerg import waImergDataset, waImergDataset_withMeta
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info

from servir.methods.ConvLSTM import ConvLSTM


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(f'./configs/{dataset_name}', f'{method_name}.py') 

config = load_config(config_path)
# log config
logging_config_info(config)

##==================Setup============================##
# Results base path for logging, working dirs, etc. 
base_results_path = f'./results/{dataset_name}'
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

# logging setup
logging_setup(base_results_path, fname=f'{method_name}.log')   
# log env info
logging_env_info()

# Setup Working dirs
work_dir = os.path.join(base_results_path, 'work_dir')
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

config['work_dir'] = work_dir   


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join('./data', dataset_name)
fname = os.path.join(dataPath, 'imerg_2020_july.h5py')

# training data from 2020-07-01 to 2020-07-21 
trainSet = waImergDataset(fname, start_date = '2020-07-01', end_date = '2020-07-08',
                           in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])
# validation data from 2020-07-22 to 2020-07-28
valSet = waImergDataset(fname, start_date = '2020-07-08', end_date = '2020-07-10',
                           in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])

# testing data from 2020-07-29 to 2020-07-31, meta data is included for saving results
testSet = waImergDataset_withMeta(fname, start_date = '2020-07-10', end_date = '2020-07-12',
                                in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])


dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=True, pin_memory=True) 
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   


# update config
config['steps_per_epoch'] = len(dataloader_train)
##==================Setup Method=====================##
# get device
if config['use_gpu']:  
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)

# log method info
logging_method_info(config, method, device)

##==============Distribution=========================##

# setup distribution
config['rank'], config['world_size'] = get_dist_info()

##==================Training=========================##
# path and name of best model
para_dict_fpath = os.path.join(base_results_path, 'model_params.pth')

if not os.path.exists(para_dict_fpath):
    train(dataloader_train, dataloader_val, method, config, para_dict_fpath)    
##==================Testing==========================## 

# Loads best model’s parameter dictionary 
method.model.load_state_dict(torch.load(para_dict_fpath))

test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, 'test_predictions.h5py'),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

# # save meta data to csv file
# test_meta_df = pd.DataFrame(test_meta, columns=['out_datetimes'])
# test_meta_df.to_csv(os.path.join(base_results_path, 'test_meta.csv'), index=True)


print("stop for debugging")


            
    










