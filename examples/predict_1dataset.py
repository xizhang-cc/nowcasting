import os
import sys
base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #
sys.path.append(base_path)

import h5py 
import torch


from servir.core.distribution import get_dist_info
from servir.datasets.dataLoader_wa_imerg import waImergDataset_withMeta
from servir.utils.config_utils import load_config
from servir.methods.ConvLSTM import ConvLSTM
from servir.utils.main_utils import print_log


#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'
data_fname = 'wa_imerg.h5'

normalize_method = '01range'
channel_sep = True
relu_last = True
loss='CFSSS'

test_st = '2020-08-25' 
test_ed = '2020-09-01'

# file names
base_fname = 'imerg_only_cfsss_log'
model_para_fname = f'{base_fname}_params.pth'
checkpoint_fname = f'{base_fname}_checkpoint.pth'
pred_fname = f'{base_fname}_predictions.h5'

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(base_path, f'configs/{dataset_name}', f'{method_name}.py') 

if os.path.isfile(config_path):
    print('config file found')
else:
    print(f'config file NOT found! config_path = {config_path}')

config = load_config(config_path)
config['loss'] = loss
config['channel_sep'] = channel_sep
config['relu_last'] = relu_last 

print_log(f'config file at {config_path} logged')

# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
    checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 
    pred_fname = pred_fname.split('.')[0] + '_local.h5'

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


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', dataset_name)
fname = os.path.join(dataPath, data_fname)


# testing data from 2020-08-25 to 2020-09-01, meta data is included for saving results
testSet = waImergDataset_withMeta(fname, start_date = test_st, end_date = test_ed,\
                                in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'], \
                                normalize_method=normalize_method)


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

##==================Testing==========================## 
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, model_para_fname)
# Loads best model’s parameter dictionary 
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))

test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

with h5py.File(fname, 'r') as hf:
    mean = hf['mean'][()]   
    std = hf['std'][()]
    max_value = hf['max'][()]
    min_value = hf['min'][()]
    
# imerg convert to mm/hr (need to be updated)
if normalize_method == 'gaussian':
    test_pred = test_pred * std + mean
elif normalize_method == '01range':
    test_pred = test_pred * (max_value - min_value) + min_value
else:
    test_pred = test_pred 


# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

print(f"PREDICTION DONE! Prediction file saved at {os.path.join(base_results_path, pred_fname)}")

            