import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")


import pandas as pd
import torch

from servir.datasets.dataLoader_EF5 import create_sample_datasets, EF5Dataset, write_forcasts_to_geotiff
from servir.extrapolation_exp import forcasts_and_save

# where to load data
dataPath = "/home/cc/projects/nowcasting/data/EF5"

# event names
EF5_events = ["Côte d'Ivoire_2018_06_19", "Côte d'Ivoire_2020_06_25", 'Ghana_2020_10_10', \
              'Ghana_2023_03_07', 'Nigeria_2020_06_18']
# 
train_st_datetimes = [list(pd.date_range(start='2018-06-18 13:00', end='2018-06-19 15:00', freq='1h')), \
                      list(pd.date_range(start='2020-06-24 16:00', end='2020-06-25 11:00', freq='1h')), \
                      list(pd.date_range(start='2020-10-09 18:00', end='2020-10-10 06:00', freq='1h')), \
                      list(pd.date_range(start='2023-03-06 10:00', end='2023-03-07 02:00', freq='1h')), \
                      list(pd.date_range(start='2020-06-17 11:00', end='2020-06-18 10:00', freq='1h'))]
                      
# 
train_len = 12
prediction_steps = 12

if not os.path.exists(os.path.join(dataPath,'EF5_samples.h5py')):
    create_sample_datasets(dataPath, EF5_events, train_st_datetimes, train_len, prediction_steps)


input_fPath = os.path.join(dataPath,'EF5_samples.h5py')
input_meta_fPath = os.path.join(dataPath,'EF5_samples_meta.csv')    


## Load data using Pytorch DataLoader
ef5_samples = EF5Dataset(input_fPath, input_meta_fPath)
dataloader = torch.utils.data.DataLoader(ef5_samples, batch_size=1, shuffle=False, pin_memory=True)


# model_config = {
#     'method': 'LINDA',
#     'max_num_features': 15,
#     'add_perturbations': False
# }

model_config = {
    'method': 'STEPS',
    'n_ens_members': 20,
    'n_cascade_levels': 6
}

# model_config = {        
#     'method': 'Lagrangian_Persistence',
# }


write2geotiff = True

# where to save results
resultsPath = "/home/cc/projects/nowcasting/examples/results/EF5"
method = model_config['method']
output_fPath = os.path.join(resultsPath,f'{method}_EF5_forcasts.h5py')
output_meta_fPath = os.path.join(resultsPath,f'{method}_EF5_forcasts_meta.csv')

forcasts, forcasts_meta = forcasts_and_save(dataloader, model_config, output_fPath, output_meta_fPath, save=True)


# write into giotiff files
if write2geotiff:
    write_forcasts_to_geotiff(output_fPath, output_meta_fPath, resultsPath, model_config)


# with h5py.File(output_fPath,'r') as hf:
#     forcasts = hf['forcasts'][:]


# forcasts.shape
print('stop for debugging')

# forcasts_meta = pd.read_csv(output_meta_fPath, index_col=0)


# print("stop for debugging") 

            
    










