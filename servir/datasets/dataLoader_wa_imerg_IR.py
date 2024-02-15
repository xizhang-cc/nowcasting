import os
import sys
base_path ='/home/cc/projects/nowcasting' # "/home1/zhang2012/nowcasting/" #
sys.path.append(base_path)

import datetime
import h5py
import numpy as np

from torch.utils.data import Dataset

from servir.datasets.dataLoader_wa_imerg import load_wa_imerg_data_from_h5
from servir.datasets.dataLoader_wa_IR import load_IR_data_from_h5    


def load_pred_IR_data_from_h5(pred_IRs_fPath):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """


    with h5py.File(pred_IRs_fPath, 'r') as hf:
        pred_IRs = hf['IRs'][:]
        IR_metas = hf['timestamps'][:]
        pred_IR_metas = [x.decode('utf-8').split(',') for x in IR_metas]


    return pred_IRs, pred_IR_metas



class waImergIRDataset(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, pred_IR_fPath, start_date, end_date, in_seq_length, out_seq_length, pred_IR_length, \
                  max_rainfall_intensity=60.0, imerg_normalize=False, IR_normalize=True, max_temp_in_kelvin=337.0):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std = load_wa_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)
        self.pred_IRs, self.pred_IR_metas = load_pred_IR_data_from_h5(pred_IR_fPath)
        # convert self.pred_IR_metas to list of datetime
        self.pred_IR_metas = [[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in y] for y in self.pred_IR_metas]

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    
        self.pred_IR_length = pred_IR_length

        # normalize the data
        if imerg_normalize:
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        else:
            self.imergs =  self.imergs / max_rainfall_intensity

                # normalize the data
        if IR_normalize:
            self.IRs = (self.IRs - self.IR_mean)/self.IR_std
        else:
            self.IRs =  self.IRs / max_temp_in_kelvin

        # crop images
        self.imergs = self.imergs[:, :, 1:-1]
        self.IRs = self.IRs[:, :, 1:-1]


    def __len__(self):
        return self.imergs.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)

        # metadata for a sample
        X_dt = list(self.imerg_dts[in_ind])
        Y_dt = list(self.imerg_dts[out_ind])

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imergs = self.imergs[in_ind]
        out_imergs = self.imergs[out_ind]

        in_imergs = np.expand_dims(in_imergs, axis=(1))
        out_imergs = np.expand_dims(out_imergs, axis=(1))

        # find the correpsonding true IR images index
        IRs_true_dt = X_dt + Y_dt[:-self.pred_IR_length]
        IRs_true_ind = [list(self.IR_dts).index(x) for x in IRs_true_dt]
        IRs_true = self.IRs[IRs_true_ind]

        # find the correpsonding pred IR images index
        IR_preds_dt = Y_dt[-self.pred_IR_length:] 
        first_pred_IR_dt = IR_preds_dt[0]
        for k, meta in enumerate(self.pred_IR_metas):
            if meta[0] == first_pred_IR_dt:
                pred_IRs_ind = k
                break
        IR_preds = self.pred_IRs[pred_IRs_ind][0::2]

        IRs = np.concatenate([IRs_true, IR_preds], axis=0)

        in_IRs = IRs[:self.in_seq_length]
        out_IRs =IRs[self.in_seq_length:self.in_seq_length+self.out_seq_length]

        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=1)
        Y = np.concatenate([out_imergs, out_IRs], axis=1)

    
        return X, Y


class waImergIRDataset_withMeta(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, pred_IR_fPath, start_date, end_date, in_seq_length, out_seq_length, pred_IR_length, \
                 imerg_normalize=False, IR_normalize=True):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std, self.imerg_max = load_wa_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std, self.IR_max = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)
        self.pred_IRs, self.pred_IR_metas = load_pred_IR_data_from_h5(pred_IR_fPath)
        # convert self.pred_IR_metas to list of datetime
        self.pred_IR_metas = [[datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in y] for y in self.pred_IR_metas]

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    
        self.pred_IR_length = pred_IR_length

        # normalize the data
        if imerg_normalize:
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        else:
            self.imergs =  self.imergs / self.imerg_max

                # normalize the data
        if IR_normalize:
            self.IRs = (self.IRs - self.IR_mean)/self.IR_std
        else:
            self.IRs =  self.IRs / self.IR_max

    def __len__(self):
        return self.imergs.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)

        # metadata for a sample
        X_dt = list(self.imerg_dts[in_ind])
        Y_dt = list(self.imerg_dts[out_ind])

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imergs = self.imergs[in_ind]
        out_imergs = self.imergs[out_ind]

        in_imergs = np.expand_dims(in_imergs, axis=(1))
        out_imergs = np.expand_dims(out_imergs, axis=(1))

        # find the correpsonding true IR images index
        IRs_true_dt = X_dt + Y_dt[:-self.pred_IR_length]
        IRs_true_ind = [list(self.IR_dts).index(x) for x in IRs_true_dt]
        IRs_true = self.IRs[IRs_true_ind]

        # find the correpsonding pred IR images index
        IR_preds_dt = Y_dt[-self.pred_IR_length:] 
        first_pred_IR_dt = IR_preds_dt[0]
        for k, meta in enumerate(self.pred_IR_metas):
            if meta[0] == first_pred_IR_dt:
                pred_IRs_ind = k
                break
        IR_preds = self.pred_IRs[pred_IRs_ind][0::2]

        IRs = np.concatenate([IRs_true, IR_preds], axis=0)

        in_IRs = IRs[:self.in_seq_length]
        out_IRs =IRs[self.in_seq_length:self.in_seq_length+self.out_seq_length]

        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=1)
        Y = np.concatenate([out_imergs, out_IRs], axis=1)

        
        # metadata for a sample
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in X_dt] 
        X_dt_str = ','.join(X_str)


        Y_dt_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in Y_dt]
        Y_dt_str = ','.join(Y_dt_str)

        return (X, Y, X_dt_str, Y_dt_str)








#===================================================================================================
#===================================================================================================
#===================================================================================================



class waImergIRDatasetTr(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, start_date, end_date, in_seq_length, out_seq_length, \
                imerg_normalize=False, IR_normalize=True):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std, self.imerg_max = load_wa_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std, self.IR_max = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if imerg_normalize:
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        else:
            self.imergs =  self.imergs / self.imerg_max 

                # normalize the data
        if IR_normalize:
            self.IRs = (self.IRs - self.IR_mean)/self.IR_std
        else:
            self.IRs =  self.IRs / self.IR_max

    def __len__(self):
        return self.imergs.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)

        # metadata for a sample
        X_dt = list(self.imerg_dts[in_ind])
        Y_dt = list(self.imerg_dts[out_ind])

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imergs = self.imergs[in_ind]
        out_imergs = self.imergs[out_ind]

        in_imergs = np.expand_dims(in_imergs, axis=(1))
        out_imergs = np.expand_dims(out_imergs, axis=(1))

        # find the correpsonding true IR images index
        in_IRs_ind = [list(self.IR_dts).index(x) for x in X_dt]
        out_IRs_ind = [list(self.IR_dts).index(x) for x in Y_dt]

        in_IRs = self.IRs[in_IRs_ind]
        out_IRs = self.IRs[out_IRs_ind]

        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=1)
        Y = np.concatenate([out_imergs, out_IRs], axis=1)

    
        return X, Y


class waImergIRDatasetTr_withMeta(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, start_date, end_date, in_seq_length, out_seq_length, imerg_normalize=False, IR_normalize=True):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std, self.imerg_max = load_wa_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std, self.IR_max = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if imerg_normalize:
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        else:
            self.imergs =  self.imergs / self.imerg_max

                # normalize the data
        if IR_normalize:
            self.IRs = (self.IRs - self.IR_mean)/self.IR_std
        else:
            self.IRs =  self.IRs / self.IR_max

    def __len__(self):
        return self.imergs.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)

        # metadata for a sample
        X_dt = list(self.imerg_dts[in_ind])
        Y_dt = list(self.imerg_dts[out_ind])

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imergs = self.imergs[in_ind]
        out_imergs = self.imergs[out_ind]

        in_imergs = np.expand_dims(in_imergs, axis=(1))
        out_imergs = np.expand_dims(out_imergs, axis=(1))

        # find the correpsonding true IR images index
        in_IRs_ind = [list(self.IR_dts).index(x) for x in X_dt]
        out_IRs_ind = [list(self.IR_dts).index(x) for x in Y_dt]

        in_IRs = self.IRs[in_IRs_ind]
        out_IRs = self.IRs[out_IRs_ind]
        
        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=1)
        Y = np.concatenate([out_imergs, out_IRs], axis=1)


        # metadata for a sample
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in X_dt] 
        X_dt_str = ','.join(X_str)


        Y_dt_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in Y_dt]
        Y_dt_str = ','.join(Y_dt_str)

        return (X, Y, X_dt_str, Y_dt_str)




#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    
    dataset1_name = 'wa_imerg'
    dataset2_name = 'wa_IR'

    data1_fname = 'wa_imerg.h5'
    data2_fname = 'wa_IR_08_m.h5'

    # where to load data
    f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
    f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

    train_st = '2020-08-25' 
    train_ed = '2020-08-28'
    val_st = '2020-08-28'
    val_ed = '2020-08-30' 
    test_st = '2020-08-30' 
    test_ed = '2020-09-01'



    a = waImergIRDatasetTr(f1name, f2name, train_st, train_ed, 12, 12)
    a.__getitem__(0)

    print('stop for debugging')




    


        
