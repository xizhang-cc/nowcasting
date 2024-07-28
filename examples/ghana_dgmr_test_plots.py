import os
import numpy as np
import torch
import datetime

from servir.methods.dgmr.dgmr import DGMR
from servir.utils import load_imerg_data_from_h5
from servir.utils import load_IR_data_from_h5
from servir.gif_creation import create_precipitation_gif

method_name = 'dgmr'

# data module
test_st = '2016-10-01 00:00:00' 
test_ed = '2016-10-31 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = 'gaussian'
use_gpu = True


metrics = ['mse', 'l1']

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#

imerg_fPath = os.path.join(os.path.join(base_path, 'data', 'ghana_imerg'), 'ghana_imerg_2011_2020_oct.h5')
IR_fPath = os.path.join(os.path.join(base_path, 'data', 'ghana_IR'), 'ghana_IR_2011_2020_oct.h5')
 

##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

result_path1 = os.path.join(base_path, 'results', 'ghana_imerg', method_name)
imerg_checkpoint_fname = os.path.join(result_path1,  f'ghana_imerg_{method_name}-{normalize_method}.ckpt') #os.path.join(result_path1, 'imerg_last.ckpt') #
# setup method
model_imerg_only = DGMR.load_from_checkpoint(imerg_checkpoint_fname)
# disable randomness, dropout, etc...
model_imerg_only.eval()

result_path2 = os.path.join(base_path, 'results', 'ghana_imerg_IR', method_name)
imerg_IR_checkpoint_fname = os.path.join(result_path2,  f'ghana_imerg_IR-{method_name}-{normalize_method}.ckpt') #os.path.join(result_path2,'imerg_withIR_last.ckpt') #
# setup method
model_imerg_withIR = DGMR.load_from_checkpoint(imerg_IR_checkpoint_fname)
# disable randomness, dropout, etc...
model_imerg_withIR.eval()


# get the sample and predict
imergs, imergs_dts, imergs_mean, imergs_std, imergs_max, imergs_min = load_imerg_data_from_h5(imerg_fPath, start_date= test_st, end_date=test_ed)
IRs, IR_dts, IR_mean, IR_std, IR_max, IR_min = load_IR_data_from_h5(IR_fPath, start_date= test_st, end_date=test_ed)

# normalize the data
if normalize_method == 'gaussian':
    imergs = (imergs - imergs_mean) / imergs_std
    IRs = -(IRs - IR_mean) / IR_std
elif normalize_method == '01range':
    imergs =  (imergs - imergs_min) / (imergs_max - imergs_min)   
    IRs =  1 - (IRs - IR_min) / (IR_max - IR_min)

# plot one sample
sample_sdt = datetime.datetime(2016, 10, 19, 11, 0)

in_dts = [sample_sdt + datetime.timedelta(minutes=30*k) for k in range(in_seq_length)]
out_dts = [sample_sdt + datetime.timedelta(minutes=30*k) for k in range(in_seq_length, in_seq_length+out_seq_length)]

# get all the index 
in_images_index = [list(imergs_dts).index(ind) for ind in in_dts]
out_images_index = [list(imergs_dts).index(ind) for ind in out_dts]


in_images = imergs[in_images_index]
out_images = imergs[out_images_index]

# get the corresponding IR data
IR_index = [list(IR_dts).index(ind) for ind in in_dts]
in_IR = IRs[IR_index]

in_images_withIRs = np.concatenate([in_images, in_IR], axis=0)

# data statistics for metrics calculation
imerg_mean = 0.12281079
imerg_std = 0.6953522
imerg_max = 53.2
imerg_min = 0.0

# expand in_images to [B, T, C, H, W]
imerg_in_images = np.expand_dims(in_images, axis=(0, 2))
imerg_in_images = imerg_in_images.astype('float32')
# predict with the model
imerg_in_images = torch.from_numpy(imerg_in_images).to(device)    
pred_out_images_imerg_only = model_imerg_only(imerg_in_images)


in_images_withIRs = np.expand_dims(in_images_withIRs, axis=(0, 2))
in_images_withIRs = in_images_withIRs.astype('float32')
in_images_withIRs = torch.from_numpy(in_images_withIRs).to(device)    
pred_out_images_imerg_IR = model_imerg_withIR(in_images_withIRs)

# move to cpu and convert to numpy array
pred_out_images_imerg_only = pred_out_images_imerg_only.cpu().detach().numpy()
pred_out_images_imerg_IR = pred_out_images_imerg_IR.cpu().detach().numpy()
# squeeze the batch dimension and the channel dimension
pred_out_images_imerg_only = np.squeeze(pred_out_images_imerg_only)
pred_out_images_imerg_IR = np.squeeze(pred_out_images_imerg_IR)

# # # normalize the data to 0 mean and 1 std
# pred_out_images_imerg_only = (pred_out_images_imerg_only - np.mean(pred_out_images_imerg_only))/np.std(pred_out_images_imerg_only)
# pred_out_images_imerg_IR = (pred_out_images_imerg_IR - np.mean(pred_out_images_imerg_IR))/np.std(pred_out_images_imerg_IR)

# change back to original scale
if normalize_method == '01range':
    pred_out_images_imerg_only = pred_out_images_imerg_only * (imerg_max - imerg_min) + imerg_min
    pred_out_images_imerg_IR = pred_out_images_imerg_IR * (imerg_max - imerg_min) + imerg_min
    out_images = out_images * (imerg_max - imerg_min) + imerg_min
elif normalize_method == 'gaussian':
    pred_out_images_imerg_only = pred_out_images_imerg_only * imerg_std + imerg_mean
    pred_out_images_imerg_IR = pred_out_images_imerg_IR * imerg_std + imerg_mean
    out_images = out_images * imerg_std + imerg_mean

# # cut the negative values
# pred_out_images_imerg_only = np.where(pred_out_images_imerg_only>=0, pred_out_images_imerg_only, 0)
# pred_out_images_imerg_IR = np.where(pred_out_images_imerg_IR>=0, pred_out_images_imerg_IR, 0)


# plot the images
# define 30 mins as the timestep datetime object
timestep = datetime.timedelta(minutes=30)

result_path = os.path.join(base_path, 'results')
create_precipitation_gif(pred_out_images_imerg_only, out_dts, timestep, result_path, title='imerg_pred', gif_dur = 1000,  geodata=None)
create_precipitation_gif(pred_out_images_imerg_IR, out_dts, timestep, result_path, title='imerg_IR_pred', gif_dur = 1000,  geodata=None)

create_precipitation_gif(out_images, out_dts, timestep, result_path, title='true', gif_dur = 1000,  geodata=None)
print('done')







