import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from servir.datasets.dataLoader_wa_imerg_npy import imergDataset_npy_withMeta
from servir.methods.dgmr.dgmr import DGMR
from pysteps import verification



method_name = 'dgmr'
dataset_name = 'wa_imerg'

# data module
test_st = '2020-01-01 00:00:00' 
test_ed = '2020-12-31 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = '01range'
use_gpu = True


metrics = ['mse', 'l1', 'fss']

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
dataPath = os.path.join(base_path, 'data', dataset_name)
 
result_path = os.path.join(base_path, 'results', dataset_name, method_name)

testSet = imergDataset_npy_withMeta(dataPath, test_st, test_ed, in_seq_length, out_seq_length,\
                                    normalize_method=normalize_method,img_shape = (352, 512))

# batch size should be 1
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False, pin_memory=False)   


##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

checkpoint_fname = os.path.join(result_path,  f'{method_name}-l1-{normalize_method}.ckpt')
# setup method
model = DGMR.load_from_checkpoint(checkpoint_fname)

# disable randomness, dropout, etc...
model.eval()

pred_results = []
pred_meta = []

# for FSS calculation
fss = verification.get_method("FSS")

scales = 8
threshold = 8


# predict with the model
pbar = tqdm(dataloader_test)
for batch in pbar:
    in_images, out_images, in_images_dt, out_images_dt = batch

    in_images = in_images.to(device)    

    pred_out_images = model(in_images)

    # move to cpu and convert to numpy array
    pred_out_images = pred_out_images.cpu().detach().numpy()

    # squeeze the batch dimension and the channel dimension
    pred_out_images = np.squeeze(pred_out_images)
    out_images = np.squeeze(out_images.numpy())

    # change back to original scale
    if normalize_method == '01range':
        pred_out_images = pred_out_images * 60
        out_images = out_images * 60
    

    # caclulate metrics per prediction steps
    for metric in metrics:
        if metric == 'mse':
            mse = np.mean((pred_out_images - out_images.numpy())**2, axis=(1,2))
        elif metric == 'l1':
            l1 = np.mean(np.abs(pred_out_images - out_images.numpy()), axis=(1,2))
        elif metric == 'fss':
            fss = []
            for i in range(out_seq_length):
                fss.append(fss(pred_out_images[i], out_images[i], scales, threshold))






