import os
import torch

from servir.datasets.dataLoader_imerg_from_tif import imergDataset_tif_withMeta
from servir.methods.convlstm.ConvLSTM import ConvLSTM


method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

# data module
test_st = '2020-10-04 00:00:00' 
test_ed = '2020-10-04 23:30:00' 
in_seq_length = 4
out_seq_length = 12
normalize_method = '01range'
use_gpu = True
loss='l1'

base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
dataPath = os.path.join(base_path, 'data', dataset_name)
 
result_path = os.path.join(base_path, 'results', dataset_name, method_name)

testSet = imergDataset_tif_withMeta(dataPath, test_st, test_ed, in_seq_length, out_seq_length,\
                                normalize_method=normalize_method,img_shape = (360, 516))

dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=2, shuffle=False, pin_memory=True, num_workers = 20)   


##==================Setup Method=====================##

if use_gpu and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

checkpoint_fname = os.path.join(result_path, f'{loss}_loss--{normalize_method}.ckpt')
# setup method
model = ConvLSTM.load_from_checkpoint(checkpoint_fname)

# disable randomness, dropout, etc...
model.eval()

# predict with the model
for batch in dataloader_test:
    batch_x, batch_y, batch_x_dt, batch_y_dt = batch
    images = torch.cat([batch_x, batch_y], dim=1)
    
    # move to device and predict
    images = images.to(device)
    pred_images,_ = model(images)

    # move to cpu and convert to numpy array
    pred_images = pred_images.cpu().detach().numpy()




# ##==================Testing==========================## 
# # # path and name of best model
# para_dict_fpath = os.path.join(base_results_path, model_para_fname)
# # Loads best model’s parameter dictionary 
# if device.type == 'cpu':
#     method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
# else:
#     method.model.load_state_dict(torch.load(para_dict_fpath))

# test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

# # save results to h5py file
# with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
#     hf.create_dataset('precipitations', data=test_pred)
#     hf.create_dataset('timestamps', data=test_meta)

# with h5py.File(fname, 'r') as hf:
#     mean = hf['mean'][()]   
#     std = hf['std'][()]
#     max_value = hf['max'][()]
#     min_value = hf['min'][()]
    

# threshold=0.1

# # imerg convert to mm/hr (need to be updated)
# if normalize_method == 'gaussian':
#     test_pred = test_pred * std + mean
# elif normalize_method == '01range':
#     test_pred = test_pred * (max_value - min_value) + min_value
# elif normalize_method == 'log_norm':
#     test_pred = np.where(test_pred < np.log10(threshold), 0.0, np.power(10, test_pred))


# # save results to h5py file
# with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
#     hf.create_dataset('precipitations', data=test_pred)
#     hf.create_dataset('timestamps', data=test_meta)

# print(f"PREDICTION DONE! Prediction file saved at {os.path.join(base_results_path, pred_fname)}")

            