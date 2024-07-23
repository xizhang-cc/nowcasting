import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split

from servir.utils import load_imerg_data_from_h5
from servir.utils import load_IR_data_from_h5    



class ghanaImergIRDataset(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, start_date, end_date, in_seq_length, out_seq_length, \
                imerg_normalize_method='gaussian', IR_normalize_method='gaussian',\
                IR_sparse = False, IR_threshold=240, time_delta = np.timedelta64(30, 'm')):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std, self.imerg_max, self.imerg_min = load_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std, self.IR_max, self.IR_min = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if imerg_normalize_method == 'gaussian':
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        elif imerg_normalize_method == '01range':
            self.imergs =  (self.imergs - self.imerg_min) / (self.imerg_max - self.imerg_min)

        if IR_normalize_method == 'gaussian':
            self.IRs = -(self.IRs - self.IR_mean)/self.IR_std
            # self.IR_threshold = -(IR_threshold - self.IR_mean)/self.IR_std
        elif IR_normalize_method == '01range':
            self.IRs =  1- (self.IRs - self.IR_min) / (self.IR_max - self.IR_min)
            # self.IR_threshold =  1 - (IR_threshold - self.IR_min) / (self.IR_max - self.IR_min)

        # if IR_sparse:
        #     self.IRs = np.where(self.IRs>=self.IR_threshold, self.IRs, 0) 

        # validate if the time delta is correct, i.e., the timesteps are continuous
        validation = np.diff(self.imerg_dts).astype('timedelta64[m]') == time_delta
        if not validation.all():
            # if not consecutive, find the index of the first non-consecutive element
            ind = np.where(~validation)[0]
            # break into list of consecutive time steps
            self.imergs = np.split(self.imergs, ind+1)
            self.imerg_dts = np.split(self.imerg_dts, ind+1)
        else:
            self.imergs = np.array([self.imergs])
            self.imerg_dts = np.array([self.imerg_dts]) 

        slen = 0
        ind_list = []
        for s in self.imerg_dts:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list
        self.slen = slen


    def __len__(self):
        return self.slen

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
        # get the index of which sequence the sample belongs to
        for i, ind in enumerate(self.ind_list):
            if idx < ind:
                break

        new_idx = idx - self.ind_list[i-1] if i > 0 else idx  
        curr_precipitations = self.imergs[i]

        in_ind = range(new_idx, new_idx+self.in_seq_length)
        out_ind = range(new_idx+self.in_seq_length, new_idx+self.in_seq_length+self.out_seq_length)

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = curr_precipitations[in_ind]
        out_imgs = curr_precipitations[out_ind]

        # desired shape: [T, C, H, W]
        in_imergs = np.expand_dims(in_imgs, axis=(1))
        out_imergs = np.expand_dims(out_imgs, axis=(1))

        in_dts = self.imerg_dts[i][in_ind]
        # out_dts = self.imerg_dts[i][out_ind]

        # get the corresponding IRs for each imerg
        in_IRs_ind = [list(self.IR_dts).index(t) for t in in_dts]
        # out_IRs_ind = [list(self.IR_dts).index(t) for t in out_dts]

        in_IRs = self.IRs[in_IRs_ind]
        # out_IRs = self.IRs[out_IRs_ind]

        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        # out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=0)
        Y = out_imergs
        # Y = np.concatenate([out_imergs, out_IRs], axis=0)


        return X, Y


class ghanaImergIRDataset_withMeta(Dataset):
    def __init__(self, imerg_fPath, IR_fPath, start_date, end_date, in_seq_length, out_seq_length, \
                imerg_normalize_method='01range', IR_normalize_method='01range',\
                IR_sparse = True, IR_threshold=240, time_delta = np.timedelta64(30, 'm')):

        self.imergs, self.imerg_dts, self.imerg_mean, self.imerg_std, self.imerg_max, self.imerg_min = load_imerg_data_from_h5(imerg_fPath,start_date= start_date, end_date=end_date)
        self.IRs, self.IR_dts, self.IR_mean, self.IR_std, self.IR_max, self.IR_min = load_IR_data_from_h5(IR_fPath, start_date= start_date, end_date=end_date)

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if imerg_normalize_method == 'gaussian':
            self.imergs = (self.imergs - self.imerg_mean)/self.imerg_std
        elif imerg_normalize_method == '01range':
            self.imergs =  (self.imergs - self.imerg_min) / (self.imerg_max - self.imerg_min)

        if IR_normalize_method == 'gaussian':
            self.IRs = -(self.IRs - self.IR_mean)/self.IR_std
            self.IR_threshold = -(IR_threshold - self.IR_mean)/self.IR_std
        elif IR_normalize_method == '01range':
            self.IRs =  1- (self.IRs - self.IR_min) / (self.IR_max - self.IR_min)
            self.IR_threshold =  1 - (IR_threshold - self.IR_min) / (self.IR_max - self.IR_min)

        if IR_sparse:
            self.IRs = np.where(self.IRs>=self.IR_threshold, self.IRs, 0) 

        # validate if the time delta is correct, i.e., the timesteps are continuous
        validation = np.diff(self.imerg_dts).astype('timedelta64[m]') == time_delta
        if not validation.all():
            # if not consecutive, find the index of the first non-consecutive element
            ind = np.where(~validation)[0]
            # break into list of consecutive time steps
            self.imergs = np.split(self.imergs, ind+1)
            self.imerg_dts = np.split(self.imerg_dts, ind+1)
        else:
            self.imergs = np.array([self.imergs])
            self.imerg_dts = np.array([self.imerg_dts]) 

        slen = 0
        ind_list = []
        for s in self.imerg_dts:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list
        self.slen = slen


    def __len__(self):
        return self.slen

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
        # get the index of which sequence the sample belongs to
        for i, ind in enumerate(self.ind_list):
            if idx < ind:
                break

        new_idx = idx - self.ind_list[i-1] if i > 0 else idx  
        curr_precipitations = self.imergs[i]

        in_ind = range(new_idx, new_idx+self.in_seq_length)
        out_ind = range(new_idx+self.in_seq_length, new_idx+self.in_seq_length+self.out_seq_length)

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = curr_precipitations[in_ind]
        out_imgs = curr_precipitations[out_ind]

        # desired shape: [T, C, H, W]
        in_imergs = np.expand_dims(in_imgs, axis=(1))
        out_imergs = np.expand_dims(out_imgs, axis=(1))

        in_dts = self.imerg_dts[i][in_ind]
        out_dts = self.imerg_dts[i][out_ind]

        # get the corresponding IRs for each imerg
        in_IRs_ind = [list(self.IR_dts).index(t) for t in in_dts]
        out_IRs_ind = [list(self.IR_dts).index(t) for t in out_dts]

        in_IRs = self.IRs[in_IRs_ind]
        out_IRs = self.IRs[out_IRs_ind]

        # # desired shape: [T, C, H, W]
        in_IRs = np.expand_dims(in_IRs, axis=(1))
        out_IRs = np.expand_dims(out_IRs, axis=(1))

        X = np.concatenate([in_imergs, in_IRs], axis=0)
        Y = np.concatenate([out_imergs, out_IRs], axis=0)

        # metadata for a sample
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in in_dts] 
        X_dt_str = ','.join(X_str)

        Y_dt_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in out_dts]
        Y_dt_str = ','.join(Y_dt_str)

        return (X, Y, X_dt_str, Y_dt_str)


class ghanaImergIRDataModule(LightningDataModule):
   

    def __init__(
        self,
        f1name: str = "/home/cc/projects/nowcasting/data/ghana_imerg/ghana_imerg_2011_2020_oct.h5",
        f2name: str = "/home/cc/projects/nowcasting/data/ghana_IR/ghana_IR_2011_2020_oct.h5",
        train_start_date: str = '2011-10-01 00:00:00',
        train_end_date: str = '2018-10-31 23:30:00',
        val_start_date: str = '2019-10-01 00:00:00',
        val_end_date: str = '2019-10-31 23:30:00',

        in_seq_length: int = 4,
        out_seq_length: int = 12,
        imerg_normalize_method: str = 'gaussian',
        IR_normalize_method: str = 'gaussian',
        IR_sparse: bool = True,
        IR_threshold: int = 240,


        batch_size: int = 12,
        shuffle: bool=False, # shuffle must set to False when using recurrent models
        pin_memory: bool=False,
    ):
        super().__init__()

        self.imergTrain = ghanaImergIRDataset(f1name, f2name, train_start_date, train_end_date, \
                                            in_seq_length, out_seq_length,\
                                            imerg_normalize_method=imerg_normalize_method, IR_normalize_method=IR_normalize_method,\
                                            IR_sparse = IR_sparse, IR_threshold=IR_threshold, time_delta = np.timedelta64(30, 'm'))
        
        self.imergVal = ghanaImergIRDataset(f1name, f2name, val_start_date, val_end_date, \
                                            in_seq_length, out_seq_length,\
                                            imerg_normalize_method=imerg_normalize_method, IR_normalize_method=IR_normalize_method,\
                                            IR_sparse = IR_sparse, IR_threshold=IR_threshold, time_delta = np.timedelta64(30, 'm'))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory


    def train_dataloader(self):
        return DataLoader(self.imergTrain, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.imergVal, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)
    
    # def test_dataloader(self):
    #     return DataLoader(self.imergTest, batch_size=self.batch_size, pin_memory=True, shuffle=self.shuffle, num_workers=20)




class ghanaImergIRRSDataModule(LightningDataModule):
   

    def __init__(
        self,
        f1name: str = "/home/cc/projects/nowcasting/data/ghana_imerg/ghana_imerg_2011_2020_oct.h5",
        f2name: str = "/home/cc/projects/nowcasting/data/ghana_IR/ghana_IR_2011_2020_oct.h5",
        train_start_date: str = '2011-10-01 00:00:00',
        train_end_date: str = '2019-10-31 23:30:00',

        in_seq_length: int = 4,
        out_seq_length: int = 12,
        imerg_normalize_method: str = 'gaussian',
        IR_normalize_method: str = 'gaussian',
        IR_sparse: bool = True,
        IR_threshold: int = 240,
        train_val_split: list=[0.9, 0.1],

        batch_size: int = 12,
        shuffle: bool=False, # shuffle must set to False when using recurrent models
        pin_memory: bool=False,
    ):
        super().__init__()

        self.imergFull = ghanaImergIRDataset(f1name, f2name, train_start_date, train_end_date, \
                                            in_seq_length, out_seq_length,\
                                            imerg_normalize_method=imerg_normalize_method, IR_normalize_method=IR_normalize_method,\
                                            IR_sparse = IR_sparse, IR_threshold=IR_threshold, time_delta = np.timedelta64(30, 'm'))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        imergFull = self.imergFull
        self.imergTrain, self.imergVal = random_split(
            imergFull, self.train_val_split, generator=torch.Generator().manual_seed(42)
        )


    def train_dataloader(self):
        return DataLoader(self.imergTrain, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.imergVal, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)
    



#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    import os 
    dataset1_name = 'ghana_imerg'
    dataset2_name = 'ghana_IR'

    data1_fname = 'ghana_imerg_2011_2020_oct.h5'
    data2_fname = 'ghana_IR_2011_2020_oct.h5'

    base_path = "/home/cc/projects/nowcasting/"

    # where to load data
    f1name = os.path.join(base_path, 'data', dataset1_name, data1_fname)
    f2name = os.path.join(base_path, 'data', dataset2_name, data2_fname)

    train_st = '2018-10-01 00:00:00' 
    train_ed = '2019-10-31 23:30:00'

    val_st = '2020-10-01 00:00:00' 
    val_ed = '2020-10-31 23:30:00'

    a = ghanaImergIRDataModule(f1name, f2name, train_st, train_ed, val_st, val_ed, 4, 12)





    


        
