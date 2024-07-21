
import sys
import datetime
# from datetime import datetime, timedelta
import h5py
import numpy as np

from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader



# load data from h5 file
def load_imerg_data_from_h5(fPath, start_date=None, end_date=None):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """


    with h5py.File(fPath, 'r') as hf:
        precipitation = hf['precipitations'][:]
        times = hf['timestamps'][:]
        times = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in times])
        mean = hf['mean'][()]   
        std = hf['std'][()]
        max = hf['max'][()]
        min = hf['min'][()]

    if (start_date is not None) and (end_date is not None):
        st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        ind = (times>=st_dt) & (times<=end_dt)

        precipitation = precipitation[ind]
        times = times[ind]

    return precipitation, times, mean, std, max, min


class imergDataset_h5(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, time_delta = np.timedelta64(30, 'm'), normalize_method='01range'):

        self.precipitations, self.datetimes, self.mean, self.std, self.max, self.min = load_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize_method == 'gaussian':
            self.precipitations = (self.precipitations - self.mean) / self.std
        elif normalize_method == '01range':
            self.precipitations =  (self.precipitations - self.min) / (self.max - self.min)   


        # validate if the time delta is correct, i.e., the timesteps are continuous
        validation = np.diff(self.datetimes).astype('timedelta64[m]') == time_delta
        if not validation.all():
            # if not consecutive, find the index of the first non-consecutive element
            ind = np.where(~validation)[0]
            # break into list of consecutive time steps
            self.precipitations = np.split(self.precipitations, ind+1)
            self.datetimes = np.split(self.datetimes, ind+1)
        else:
            self.precipitations = np.array([self.precipitations])
            self.datetimes = np.array([self.datetimes]) 

        slen = 0
        ind_list = []
        for s in self.datetimes:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list
        self.slen = slen

    def __len__(self):

        return self.slen

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 
        
        # get the index of which sequence the sample belongs to
        for i, ind in enumerate(self.ind_list):
            if idx < ind:
                break

        new_idx = idx - self.ind_list[i-1] if i > 0 else idx  
        curr_precipitations = self.precipitations[i]

        in_ind = range(new_idx, new_idx+self.in_seq_length)
        out_ind = range(new_idx+self.in_seq_length, new_idx+self.in_seq_length+self.out_seq_length)

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = curr_precipitations[in_ind]
        out_imgs = curr_precipitations[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        return X, Y


class imergDataset_h5_withMeta(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, time_delta = np.timedelta64(30, 'm'), normalize_method='01range'):

        self.precipitations, self.datetimes, self.mean, self.std, self.max, self.min = load_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize_method == 'gaussian':
            self.precipitations = (self.precipitations - self.mean) / self.std
        elif normalize_method == '01range':
            self.precipitations =  (self.precipitations - self.min) / (self.max - self.min)   


        # validate if the time delta is correct, i.e., the timesteps are continuous
        validation = np.diff(self.datetimes).astype('timedelta64[m]') == time_delta
        if not validation.all():
            # if not consecutive, find the index of the first non-consecutive element
            ind = np.where(~validation)[0]
            # break into list of consecutive time steps
            self.precipitations = np.split(self.precipitations, ind+1)
            self.datetimes = np.split(self.datetimes, ind+1)
        else:
            self.precipitations = np.array([self.precipitations])
            self.datetimes = np.array([self.datetimes]) 

        slen = 0
        ind_list = []
        for s in self.datetimes:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list
        self.slen = slen  

    def __len__(self):
        # slen = 0
        # ind_list = []
        # for s in self.datetimes:
        #     curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
        #     ind_list.append(slen + curr_len)

        #     slen += curr_len

        # self.ind_list = ind_list

        return self.slen

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 
        
        # get the index of which sequence the sample belongs to
        for i, ind in enumerate(self.ind_list):
            if idx <= ind:
                break


        curr_precipitations = self.precipitations[i]
        curr_datetimes = self.datetimes[i]

        new_idx = idx - self.ind_list[i-1] if i > 0 else idx    


        in_ind = range(new_idx, new_idx+self.in_seq_length)
        out_ind = range(new_idx+self.in_seq_length, new_idx+self.in_seq_length+self.out_seq_length)


        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = curr_precipitations[in_ind]
        out_imgs = curr_precipitations[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        # metadata for a sample
        X_dt = curr_datetimes[in_ind]
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in X_dt] 
        X_dt_str = ','.join(X_str)


        Y_dt = curr_datetimes[out_ind]
        Y_dt_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in Y_dt]
        Y_dt_str = ','.join(Y_dt_str)

        return (X, Y, X_dt_str, Y_dt_str)


class ghanaImergDataModule(LightningDataModule):
   

    def __init__(
        self,
        fPath: str = "/home/cc/projects/nowcasting/data/ghana_imerg/ghana_imerg_2011_2020_oct.h5",
        train_start_date: str = '2011-10-01 00:00:00',
        train_end_date: str = '2018-11-01 00:00:00',
        val_start_date: str = '2019-10-01 00:00:00',
        val_end_date: str = '2019-11-01 00:00:00',

        in_seq_length: int = 4,
        out_seq_length: int = 12,
        normalize_method: str = '01range',


        batch_size: int = 12,
        shuffle: bool=False, # shuffle must set to False when using recurrent models
        pin_memory: bool=False,
    ):
        super().__init__()

        self.imergTrain = imergDataset_h5(fPath, train_start_date, train_end_date,\
                                        in_seq_length, out_seq_length,normalize_method=normalize_method)
        
        self.imergVal = imergDataset_h5(fPath, val_start_date, val_end_date,\
                                        in_seq_length, out_seq_length,normalize_method=normalize_method)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory


    def train_dataloader(self):
        return DataLoader(self.imergTrain, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.imergVal, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=self.shuffle)
    
    # def test_dataloader(self):
    #     return DataLoader(self.imergTest, batch_size=self.batch_size, pin_memory=True, shuffle=self.shuffle, num_workers=20)



#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    

    dataPath = "/home/cc/projects/nowcasting/data/ghana_imerg/"

    start_date = '2011-10-01'
    end_date = '2018-11-01' 
    fPath = dataPath+'ghana_imerg_2011_2020_oct.h5'

    a = imergDataset_h5_withMeta(fPath, start_date, end_date, 4, 12)
    a.__getitem__(11783)

    print('stop for debugging')




    


        
