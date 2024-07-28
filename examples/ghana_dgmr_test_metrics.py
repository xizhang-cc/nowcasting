import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

from servir.datasets.dataLoader_ghana_imerg_h5 import imergDataset_h5, imergDataset_h5_withMeta
from servir.methods.dgmr.dgmr import DGMR
from pysteps import verification



method_name = 'dgmr'
dataset_name = 'ghana_imerg'

# data module
test_st = '2020-10-01 00:00:00' 
test_ed = '2020-10-31 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = 'gaussian'
use_gpu = True


metrics = ['mse', 'l1']

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
dataPath = os.path.join(base_path, 'data', dataset_name)
fPath = os.path.join(dataPath, 'ghana_imerg_2011_2020_oct.h5')
 
result_path = os.path.join(base_path, 'results', dataset_name, method_name)

testSet = imergDataset_h5_withMeta(fPath, test_st, test_ed, in_seq_length, out_seq_length)

# batch size should be 1
dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=1, shuffle=False, pin_memory=False)   


##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

checkpoint_fname = os.path.join(result_path,  f'{method_name}-{normalize_method}.ckpt')
# setup method
model = DGMR.load_from_checkpoint(checkpoint_fname)

# disable randomness, dropout, etc...
model.eval()

pred_results = []
pred_meta = []

# for FSS calculation
fss = verification.get_method("FSS")

scales = 2
threshold = 1.0

# data statistics for metrics calculation
imerg_mean = 0.12281079
imerg_std = 0.6953522
imerg_max = 53.2
imerg_min = 0.0

# IR_mean = 277.7857516
# IR_std = 22.926133625876606
# IR_max = 343.15868619144055
# IR_min = 90.69283181067479

mse_scores = []
l1_scores = []
# predict with the model
pbar = tqdm(dataloader_test)
for batch in pbar:
    in_images, out_images, in_dts, out_dts = batch

    in_images = in_images.to(device)    

    pred_out_images = model(in_images)

    # move to cpu and convert to numpy array
    pred_out_images = pred_out_images.cpu().detach().numpy()
    out_images = out_images.cpu().detach().numpy()

    # squeeze the batch dimension and the channel dimension
    pred_out_images = np.squeeze(pred_out_images)
    out_images = np.squeeze(out_images)

    # change back to original scale
    if normalize_method == '01range':
        pred_out_images = pred_out_images * (imerg_max - imerg_min) + imerg_min
        out_images = out_images * (imerg_max - imerg_min) + imerg_min
    elif normalize_method == 'gaussian':
        pred_out_images = pred_out_images * imerg_std + imerg_mean
        out_images = out_images * imerg_std + imerg_mean

    # cut the negative values
    pred_out_images = np.where(pred_out_images>=0, pred_out_images, 0)

    # caclulate metrics per prediction steps

    mse = np.mean((pred_out_images - out_images)**2, axis=(1,2))
    l1 = np.mean(np.abs(pred_out_images - out_images), axis=(1,2))

    # fss_scores = []
    # for i in range(out_seq_length):
    #     fss_scores.append(fss(pred_out_images[i], out_images[i], scales, threshold))

    mse_scores.append(mse)
    l1_scores.append(l1)

# calculate the mean of the metrics
mse_scores = np.mean(np.concatenate(mse_scores), axis=0)
l1_scores = np.mean(np.concatenate(l1_scores), axis=0)

    






