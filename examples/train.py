import os
import sys
base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)

import h5py 
import time
import torch


from servir.core.distribution import get_dist_info
from servir.core.trainer import train
from servir.datasets.dataLoader_wa_imerg import waImergDataset
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info

from servir.methods.ConvLSTM import ConvLSTM


#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'
data_fname = 'wa_imerg.h5'

train_st = '2020-08-25' #'2020-06-01'
train_ed = '2020-08-28' #'2020-08-18'
val_st = '2020-08-28' #'2020-08-18'
val_ed = '2020-08-30' #'2020-08-25'

# for data normalization
max_value = 60.0
normalize = False

model_para_fname = 'imerg_only_mse_params.pth'
checkpoint_fname = 'imerg_only_mse_checkpoint.pth'


#================================================#

# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
    checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print(f'results path : {base_results_path}')

# logging setup
logging_setup(base_results_path, fname=f'{method_name}.log')   
print('logging file created')
# log env info
logging_env_info()
print('env info logged')

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)


# log config
logging_config_info(config)
print('configuration file logged')

##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', dataset_name)
fname = os.path.join(dataPath, data_fname)

# training data from 2020-06-01 to 2020-08-18 
trainSet = waImergDataset(fname, start_date = train_st, end_date = train_ed,\
                        in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'],\
                        max_rainfall_intensity = max_value, normalize=normalize)

# validation data from 2020-08-18 to 2020-08-25
valSet = waImergDataset(fname, start_date = val_st, end_date = val_ed,\
                        in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'], \
                        max_rainfall_intensity = max_value, normalize=normalize)

print('Dataset created.')
print(f'training_len = {len(trainSet)}')
print(f'val_len = {len(valSet)}')


dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=True, pin_memory=True) 

# update config
config['steps_per_epoch'] = len(dataloader_train)
##==================Setup Method=====================##
# get device
print(f'There are total {torch.cuda.device_count()} GPUs on current node')

if (config['use_gpu']) and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
    gpu = torch.cuda.get_device_properties(device)
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
print(f'model parameters saved at {para_dict_fpath}')

checkpoint_fpath = os.path.join(base_results_path, checkpoint_fname)
print(f'model training checkpoint saved at {checkpoint_fpath}')

train(dataloader_train, dataloader_val, method, config, para_dict_fpath, checkpoint_fpath)    

print("TRAINING DONE")



            
    










