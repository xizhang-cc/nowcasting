import os
import sys
base_path = "/home1/zhang2012/nowcasting/"#'/home/cc/projects/nowcasting' #
sys.path.append(base_path)

import h5py 
import torch





#================Specification=========================#
method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'

self.imergTest = imergDataset_tif(dataPath, test_start_date, test_end_date, in_seq_length, out_seq_length,\
                                sampling_freq=sampling_freq, normalize_method=normalize_method,img_shape = img_shape)

# # test run on local machine
# if base_path == '/home/cc/projects/nowcasting':
#     model_para_fname = model_para_fname.split('.')[0] + '_local.pth'
#     checkpoint_fname = checkpoint_fname.split('.')[0] + '_local.pth' 
#     pred_fname = pred_fname.split('.')[0] + '_local.h5'

#     test_st = '2020-08-30'
#     test_ed = '2020-09-01'

#     config['batch_size'] = 2
#     config['val_batch_size'] = 2
#     config['num_hidden'] = '32, 32' 
#     config['max_epoch'] = 10
#     config['early_stop_epoch'] = 2 # test run on local machine

# Results base path for logging, working dirs, etc. 
base_results_path = os.path.join(base_path, f'results/{dataset_name}')
if not os.path.exists(base_results_path):
    os.makedirs(base_results_path)

print_log(f'results path : {base_results_path}')


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', dataset_name)
fname = os.path.join(dataPath, data_fname)


# testing data from 2020-08-25 to 2020-09-01, meta data is included for saving results
testSet = imergDataset_withMeta(fname, start_date = test_st, end_date = test_ed,\
                                in_seq_length = config['in_seq_length'], out_seq_length=config['out_seq_length'], \
                                normalize_method=normalize_method)


dataloader_test = torch.utils.data.DataLoader(testSet, batch_size=config['val_batch_size'], shuffle=False, pin_memory=True)   

# update config
config['steps_per_epoch'] = 10

# setup distribution
config['rank'], config['world_size'] = get_dist_info()
##==================Setup Method=====================##

if (config['use_gpu']) and torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

config['device'] = device

# setup method
method = ConvLSTM(config)

##==================Testing==========================## 
# # path and name of best model
para_dict_fpath = os.path.join(base_results_path, model_para_fname)
# Loads best model’s parameter dictionary 
if device.type == 'cpu':
    method.model.load_state_dict(torch.load(para_dict_fpath, map_location=torch.device('cpu')))
else:
    method.model.load_state_dict(torch.load(para_dict_fpath))

test_loss, test_pred, test_meta = method.test(dataloader_test, gather_pred = True)

# save results to h5py file
with h5py.File(os.path.join(base_results_path, pred_fname),'w') as hf:
    hf.create_dataset('precipitations', data=test_pred)
    hf.create_dataset('timestamps', data=test_meta)

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

            