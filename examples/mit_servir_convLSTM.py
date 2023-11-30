import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import h5py
import logging 
import time
import numpy as np
import pandas as pd
import torch

from servir.datasets.dataloader_servirMIT import load_mit_servir_data, ServirDataset
from servir.utils.config_utils import load_config
from servir.utils.logger_utils import logging_setup, logging_env_info, logging_config_info, logging_method_info

from servir.methods.ConvLSTM import ConvLSTM


method = 'ConvLSTM'
dataname = 'mit_servir'

write2geotiff = True

# Results and Logging
base_results_path = f'./results/{dataname}'
logging_setup(base_results_path, fname=f'{method}.log')   

# log env info
logging_env_info()

##=============Read In Configurations================##
# Load configuration file
config_path = os.path.join(f'./configs/{dataname}', f'{method}.py') 
config = load_config(config_path)
# log config
logging_config_info(config)
##===================================================##

##==================Data Loading=====================##
# where to load data
dataPath = f"{config['data_root']}/{dataname}"

X_train, Y_train, X_val, Y_val, X_test, Y_test, training_meta, val_meta, testing_meta = \
load_mit_servir_data(dataPath, TRAIN_VAL_FRAC=0.8, N_TRAIN=100, N_TEST=20)

# Load data using Pytorch DataLoader
trainSet = ServirDataset(X_train, X_train)
valSet = ServirDataset(X_val, Y_val)
testSet = ServirDataset(X_test, Y_test)

dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True) 
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

##===================================================##


##==================Setup Method=====================##
# get device
if config['use_gpu']:  
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# setup method
steps_per_epoch = len(dataloader_train)
method = ConvLSTM(config, steps_per_epoch, device)

# log method info
logging_method_info(config, method, device)
##===================================================##

##==================Training=========================##



print("stop for debugging") 

            
    










