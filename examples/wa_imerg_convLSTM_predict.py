import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from servir.datasets.dataLoader_imerg_from_npy import imergDataset_npy_withMeta
from servir.methods.convlstm.ConvLSTM import ConvLSTM


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

# data module
test_st = '2020-01-01 00:00:00' 
test_ed = '2020-12-31 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = '01range'
use_gpu = True
loss='l1'


base_path = "/home1/zhang2012/nowcasting/"
dataPath = os.path.join(base_path, 'data', dataset_name)
 
result_path = os.path.join(base_path, 'results', dataset_name, method_name)

testSet = imergDataset_npy_withMeta(dataPath, test_st, test_ed, in_seq_length, out_seq_length,\
                                    normalize_method=normalize_method,img_shape = (360, 516))

dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=6, shuffle=False, pin_memory=False)   


##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

checkpoint_fname = os.path.join(result_path,  f'{method_name}-{loss}-{normalize_method}.ckpt')
# setup method
model = ConvLSTM.load_from_checkpoint(checkpoint_fname)

# disable randomness, dropout, etc...
model.eval()

pred_results = []
pred_meta = []
# predict with the model
pbar = tqdm(dataloader_test)
for batch in pbar:
    in_images, out_images, in_images_dt, out_images_dt = batch
    images = torch.cat([in_images, out_images], dim=1)
    
    # move to device and predict
    images = images.to(device)
    pred_out_images,_ = model(images)

    # move to cpu and convert to numpy array
    pred_out_images = pred_out_images.cpu().detach().numpy()

    # save the results
    pred_results.append(pred_out_images)


    pred_meta = pred_meta + [dt for dt in out_images_dt]

if len(pred_results)>0:
    pred_results = np.concatenate(pred_results, axis=0)


if pred_results.shape[2] == 1: # if grayscale, remove the channel dimension. [S, T, 1, H, W] --> [S, T, H, W]
    pred_results = np.squeeze(pred_results, axis=2)


imerg_mean: float = 0.04963324009442847
imerg_std: float = 0.5011062947027829
imerg_max: float = 60.0
imerg_min: float = 0.0

# imerg convert to mm/hr 
if normalize_method == '01range':
    test_pred = pred_results * (imerg_max - imerg_min) + imerg_min
elif normalize_method == 'norm':
    test_pred = pred_results * imerg_std + imerg_mean


pred_fname = f'{method_name}-{loss}-{normalize_method}-predictions.h5'
# save results to h5py file
with h5py.File(os.path.join(result_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=pred_meta)


print('Prediction results saved to:', os.path.join(result_path, pred_fname))

