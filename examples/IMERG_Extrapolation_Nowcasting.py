import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import h5py

import numpy as np
import pandas as pd
import torch
from pysteps import verification

from servir.datasets.dataLoader_EF5 import EF5Dataset
from servir.extrapolation_run import forcasts_and_save

# where to load data
dataPath = "/home/cc/projects/nowcasting/data/EF5"

input_fPath = os.path.join(dataPath,'EF5_samples.h5py')
input_meta_fPath = os.path.join(dataPath,'EF5_samples_meta.csv')    

# where to save results
resultsPath = "/home/cc/projects/nowcasting/examples/results/EF5"
output_fPath = os.path.join(resultsPath,'EF5_forcasts.h5py')
output_meta_fPath = os.path.join(resultsPath,'EF5_forcasts_meta.csv')

## Load data using Pytorch DataLoader
ef5_samples = EF5Dataset(input_fPath, input_meta_fPath)
dataloader = torch.utils.data.DataLoader(ef5_samples, batch_size=1, shuffle=False, pin_memory=True)

# model config
model_config = {
    'method': 'LINDA',
    'max_num_features': 15,
    'add_perturbations': False
}

forcasts_and_save(dataloader, model_config, output_fPath, output_meta_fPath)


    
    

# FSS score
# calculate FSS
fss = verification.get_method("FSS")

thr=1.0
scale=2



            
    










