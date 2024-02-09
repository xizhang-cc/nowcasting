import os
import sys
base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #
sys.path.append(base_path)

import h5py 
import time
import torch


from servir.core.distribution import get_dist_info
from servir.core.trainer import train
from servir.datasets.dataLoader_wa_imerg import waImergDataset, waImergDataset_withMeta
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info

from servir.methods.ConvLSTM import ConvLSTM


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'


# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print(f'results path : {base_results_path}')


##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', dataset_name)
fname = os.path.join(dataPath, 'wa_imerg.h5')


# testing data from 2020-08-25 to 2020-09-01, meta data is included for saving results
testSet = waImergDataset_withMeta(fname, start_date = '2020-08-25', end_date = '2020-09-01',\
                                in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'])


dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# update config
config['steps_per_epoch'] = 10
##==================Setup Method=====================##

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



##==================Testing==========================## 
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, 'imerg_only_mse_params.pth')
# Loads best model’s parameter dictionary 
method.model.load_state_dict(torch.load(para_dict_fpath))


test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, 'imerg_only_mse_predictions.h5'),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

print(f'results saved at {os.path.join(base_results_path, "imerg_only_mse_predictions.h5")}')

print("DONE")


            
    










