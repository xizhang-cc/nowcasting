import os
import sys
base_path = '/home/cc/projects/nowcasting'
sys.path.append(base_path)

import h5py 
import torch
import datetime
import numpy as np
from matplotlib import pyplot as plt

from servir.datasets.dataLoader_wa_imerg import load_wa_imerg_data_from_h5
from servir.datasets.dataLoader_wa_IR import load_IR_data_from_h5    

#================Specification=========================#
method_name = 'ConvLSTM'

dataset1_name = 'wa_imerg'
dataset2_name = 'wa_IR'

data1_fname = 'wa_imerg.h5'
data2_fname = 'wa_IR.h5'

# new data name
dataset_name = 'wa_imerg_IR'

st = '2020-08-25' 
ed = '2020-09-01'

imerg_normalize_method = '01range'
IR_normalize_method = '01range'


# test run on local machine
if base_path == '/home/cc/projects/nowcasting':
    data2_fname = 'wa_IR_08.h5'

##==================Data Loading=====================##
# where to load data
f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

imergs, imerg_dts, imerg_mean, imerg_std, imerg_max, imerg_min = load_wa_imerg_data_from_h5(f1name,start_date= st, end_date=ed)
IRs, IR_dts, IR_mean, IR_std, IR_max, IR_min = load_IR_data_from_h5(f2name, start_date= st, end_date=ed)


# normalize the data
if imerg_normalize_method == 'gaussian':
    imergs = (imergs - imerg_mean)/imerg_std
elif imerg_normalize_method == '01range':
    imergs =  (imergs - imerg_min) / (imerg_max - imerg_min)
else:
    imergs = imergs   

# normalize the data
if IR_normalize_method=='gaussian':
    IRs = (IRs - IR_mean)/IR_std
elif IR_normalize_method=='01range':
    IRs =  1 -  (IRs - IR_min) / (IR_max - IR_min)
else:
    IRs = IRs 

sample_dt = datetime.datetime(2020, 8, 25, 11, 0)
imerg_idx = list(imerg_dts).index(sample_dt)
IR_idx = list(IR_dts).index(sample_dt)

imerg = imergs[imerg_idx]
IR = IRs[IR_idx]    

plt.figure()
plt.imshow(imerg*60, cmap='gray')

plt.figure()    
plt.imshow(IR, cmap='gray')

from servir.visulizations.gif_creation import create_precipitation_plots


from pysteps.visualization import plot_precip_field
plt.figure()
plot_precip_field(imerg*60)

print('stop for debugging')




            
