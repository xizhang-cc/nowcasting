import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")

import h5py
import torch

from servir.core.distribution import get_dist_info
from servir.core.trainer import train
from servir.datasets.dataLoader_wa_imerg import waImergDataset
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
dataPath = f"{config['data_root']}/{dataset_name}"


# training data from 2020-07-01 to 2020-07-21
trainSet = waImergDataset(dataPath, start_date = '2020-07-01', end_date = '2020-07-22',
                           in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])
# validation data from 2020-07-22 to 2020-07-28
valSet = waImergDataset(dataPath, start_date = '2020-07-22', end_date = '2020-07-29',
                           in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])
# testing data from 2020-07-29 to 2020-07-31
testSet = waImergDataset(dataPath, start_date = '2020-07-29', end_date = '2020-07-31',
                           in_seq_length = config['in_seq_length'], out_seq_length=config['in_seq_length'])


dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True) 
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
# logging_method_info(config, method, device)

##==============Distribution=========================##

# setup distribution
config['rank'], config['world_size'] = get_dist_info()

##==================Training=========================##
best_model_path = train(dataloader_train, dataloader_val, method, config)

##==================Testing==========================## 
# load best model
method.model.load_state_dict(torch.load(best_model_path))

test_loss, test_pred = method.test(dataloader_val, gather_pred = True)

## save test results to h5 file

with h5py.File(os.path.join(base_results_path, f'{dataset_name}_{method_name}_predictions.h5py'),'w') as hf:
    for k in test_pred.keys():
        hf.create_dataset(k,data=test_pred[k])



# print("stop for debugging") 

            
    










