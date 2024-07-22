import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from servir.datasets.dataLoader_wa_imerg_npy import imergDataset_npy_withMeta
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

metrics = ['mse', 'l1', 'fss']

base_path = "/home1/zhang2012/nowcasting/"
dataPath = os.path.join(base_path, 'data', dataset_name)
 
result_path = os.path.join(base_path, 'results', dataset_name, method_name)

testSet = imergDataset_npy_withMeta(dataPath, test_st, test_ed, in_seq_length, out_seq_length,\
                                    normalize_method=normalize_method,img_shape = (360, 516))

# batch size should be 1
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False, pin_memory=False)   


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

    # caclulate metrics per prediction steps
    for metric in metrics:
        if metric == 'mse':
            mse = np.mean((pred_out_images - out_images.numpy())**2)
        elif metric == 'l1':
            l1 = np.mean(np.abs(pred_out_images - out_images.numpy()))
        elif metric == 'fss':
            # calculate fss
            pass





