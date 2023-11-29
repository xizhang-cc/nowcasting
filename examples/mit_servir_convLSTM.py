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

from servir.methods.ConvLSTM import ConvLSTM


method = 'ConvLSTM'
dataname = 'mit_servir'

use_gpu = True
distributed = False

# Results and Logging
base_results_path = f'./results/{dataname}'

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(base_results_path, f'{method}_{timestamp}.log'),
                    filemode='a', format='%(asctime)s - %(message)s')


logging.info('Method: ' + str(method))

write2geotiff = True

# Load configuration file
config_path = os.path.join(f'./configs/{dataname}', f'{method}.py') 
config = load_config(config_path)

# where to load data
dataPath = f"{config['data_root']}/{dataname}"

X_train, Y_train, X_val, Y_val, X_test, Y_test, training_meta, val_meta, testing_meta = \
load_mit_servir_data(dataPath, TRAIN_VAL_FRAC=0.8, N_TRAIN=100, N_TEST=20)


## Load data using Pytorch DataLoader
trainSet = ServirDataset(X_train, X_train)
valSet = ServirDataset(X_val, Y_val)
testSet = ServirDataset(X_test, Y_test)

dataloader_train = torch.utils.data.DataLoader(trainSet, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
dataloader_val = torch.utils.data.DataLoader(valSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True) 
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

#
max_iters =  config['max_epoch'] * len(dataloader_train)
steps_per_epoch = len(dataloader_train)


if config['early_stop_epoch'] <= config['max_epoch'] // 5:
     config['early_stop_epoch'] = config['max_epoch'] * 2



if config['use_gpu']:  
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

method = ConvLSTM(config, device, steps_per_epoch)


print("stop for debugging") 

            
    










