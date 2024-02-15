import os
import sys
base_path = '/home/cc/projects/nowcasting'#"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)

import h5py 
import time
import torch
import logging


from servir.core.distribution import get_dist_info
from servir.core.trainer import train
from servir.datasets.dataLoader_wa_imerg_IR import waImergIRDatasetTr, waImergIRDatasetTr_withMeta
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info
from servir.utils.main_utils import print_log

from servir.methods.ConvLSTM import ConvLSTM


#================Specification=========================#
method_name = 'ConvLSTM'

dataset1_name = 'wa_imerg'
dataset2_name = 'wa_IR'

data1_fname = 'wa_imerg.h5'
data2_fname = 'wa_IR_08_m.h5'

# new data name
dataset_name = 'wa_imerg_IR'

train_st = '2020-08-25' #'2020-06-01' #
train_ed = '2020-08-28' #'2020-08-18' #
val_st = '2020-08-28'#'2020-08-18' #
val_ed = '2020-08-30' #'2020-08-25' #
test_st = '2020-08-30' #'2020-08-25' 
test_ed = '2020-09-01'

# train_st = '2020-06-01' 
# train_ed = '2020-08-18' 
# val_st = '2020-08-18'
# val_ed = '2020-08-25'
# test_st = '2020-08-25' 
# test_ed = '2020-09-01'

# file names
base_fname = 'imerg_IR_mse'
model_para_fname = f'{base_fname}_params.pth'
checkpoint_fname = f'{base_fname}_checkpoint.pth'
pred_fname = f'{base_fname}_predictions.h5'

#================================================#

# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
    checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 
    pred_fname = pred_fname.split('.')[0] + '_local.h5'

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print_log(f'results path : {base_results_path}')

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

print_log(f'config file at {config_path} logged')


# log config
logging_config_info(config)
print('configuration file logged')

##==================Data Loading=====================##
# where to load data
f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

trainSet = waImergIRDatasetTr(f1name, f2name, train_st, train_ed, \
                        in_seq_length=config['in_seq_length'],  out_seq_length=config['out_seq_length'], \
                        max_rainfall_intensity=60.0, imerg_normalize=False, IR_normalize=True, max_temp_in_kelvin=337.0)


valSet = waImergIRDatasetTr(f1name, f2name, val_st, val_ed, \
                        in_seq_length=config['in_seq_length'],  out_seq_length=config['out_seq_length'], \
                        max_rainfall_intensity=60.0, imerg_normalize=False, IR_normalize=True, max_temp_in_kelvin=337.0)

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
    gpu = torch.cuda.get_device_properties(device)
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)


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
testSet = waImergIRDatasetTr(f1name, f2name, test_st, test_ed, \
                        in_seq_length=config['in_seq_length'],  out_seq_length=config['out_seq_length'], \
                        max_rainfall_intensity=60.0, imerg_normalize=False, IR_normalize=True, max_temp_in_kelvin=337.0)

dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# Loads best model’s parameter dictionary 
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))


test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)
if config['normalize']:
    test_pred  = test_pred * config['std'] + config['mean']
else:
    test_pred  = test_pred * config['max_value']

# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

logging.info(f"PREDICTION DONE! Prediction file saved at {os.path.join(base_results_path, pred_fname)}")
            
    









