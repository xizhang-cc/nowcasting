import os
import numpy as np
import torch
import datetime

from servir.methods.dgmr.dgmr import DGMR
from servir.utils import load_imerg_data_from_h5

method_name = 'dgmr'

# data module
test_st = '2011-10-01 00:00:00' 
test_ed = '2020-10-31 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = 'gaussian'
use_gpu = True


metrics = ['mse', 'l1']

base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #

imerg_fPath = os.path.join(os.path.join(base_path, 'data', 'ghana_imerg'), 'ghana_imerg_2011_2020_oct.h5')
 

##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

result_path1 = os.path.join(base_path, 'results', 'ghana_imerg', method_name)

fileName = 'last-42720.ckpt' # f'{method_name}-{normalize_method}-42944.ckpt' #
imerg_checkpoint_fname = os.path.join(result_path1, 'last.ckpt') #os.path.join(result_path1,  f'ghana_imerg_{method_name}-{normalize_method}.ckpt') #
# setup method
model_imerg_only = DGMR.load_from_checkpoint(imerg_checkpoint_fname)
# disable randomness, dropout, etc...
model_imerg_only.eval()


# get the sample and predict
imergs, imergs_dts, imergs_mean, imergs_std, imergs_max, imergs_min = load_imerg_data_from_h5(imerg_fPath, start_date= test_st, end_date=test_ed)

# normalize the data
if normalize_method == 'gaussian':
    imergs = (imergs - imergs_mean) / imergs_std
elif normalize_method == '01range':
    imergs =  (imergs - imergs_min) / (imergs_max - imergs_min)   

# plot one sample
sample_sdt = datetime.datetime(2020, 10, 9, 0, 0)

in_dts = [sample_sdt + datetime.timedelta(minutes=30*k) for k in range(in_seq_length)]
out_dts = [sample_sdt + datetime.timedelta(minutes=30*k) for k in range(in_seq_length, in_seq_length+out_seq_length)]

# get all the index 
in_images_index = [list(imergs_dts).index(ind) for ind in in_dts]
out_images_index = [list(imergs_dts).index(ind) for ind in out_dts]


in_images = imergs[in_images_index]
out_images = imergs[out_images_index]



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


# move to cpu and convert to numpy array
pred_out_images_imerg_only = pred_out_images_imerg_only.cpu().detach().numpy()


# change back to original scale
if normalize_method == '01range':
    pred_out_images_imerg_only = pred_out_images_imerg_only * (imerg_max - imerg_min) + imerg_min
    out_images = out_images * (imerg_max - imerg_min) + imerg_min
elif normalize_method == 'gaussian':
    pred_out_images_imerg_only = pred_out_images_imerg_only * imerg_std + imerg_mean
    out_images = out_images * imerg_std + imerg_mean


# reduce dimensions for plotting
preds = np.squeeze(pred_out_images_imerg_only)
gts = np.squeeze(out_images)

from matplotlib import pyplot as plt
from pysteps.visualization import plot_precip_field

for id in range(len(preds)):
    plt.figure()
    plot_precip_field(preds[id])
    plt.title(f'pred - {datetime.datetime.strftime(out_dts[id], "%Y-%m-%d %H:%M:%S")}')

    plt.figure()
    plot_precip_field(gts[id])
    plt.title(f'true - {datetime.datetime.strftime(out_dts[id], "%Y-%m-%d %H:%M:%S")}')


    # plt.figure()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle('Horizontally stacked subplots')
    # plot_precip_field(img)
    # plot_precip_field



# # plot the images
# # define 30 mins as the timestep datetime object
# timestep = datetime.timedelta(minutes=30)

# result_path = os.path.join(base_path, 'results')
# create_precipitation_gif(pred_out_images_imerg_only, out_dts, timestep, result_path, title='imerg_pred', gif_dur = 1000,  geodata=None)

# create_precipitation_gif(out_images, out_dts, timestep, result_path, title='true', gif_dur = 1000,  geodata=None)
# print('done')







