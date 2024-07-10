
import sys
import datetime
import h5py
import numpy as np

from torch.utils.data import Dataset



# load data from h5 file
def load_imerg_data_from_h5(fPath, start_date, end_date):
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

    st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    ind = (times>=st_dt) & (times<end_dt)

    requested_precipitation = precipitation[ind]
    requested_times = times[ind]

    return requested_precipitation, requested_times, mean, std, max, min

def imerg_log_normalize(data, threshold=0.1, zerovalue=-2.0):
    new = np.where(data < threshold, zerovalue, np.log10(data))
    return new

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
        elif normalize_method == 'log_norm':
            self.precipitations = imerg_log_normalize(self.precipitations)

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

    def __len__(self):
        slen = 0
        ind_list = []
        for s in self.datetimes:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list

        return slen

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
        elif normalize_method == 'log_norm':
            self.precipitations = imerg_log_normalize(self.precipitations)

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

    def __len__(self):
        slen = 0
        ind_list = []
        for s in self.datetimes:
            curr_len = len(s)-self.in_seq_length-self.out_seq_length+1
            ind_list.append(slen + curr_len)

            slen += curr_len

        self.ind_list = ind_list

        return slen

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



#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    

    dataPath = "/home/cc/projects/nowcasting/data/ghana_imerg/"

    start_date = '2011-10-01'
    end_date = '2018-11-01' 
    fPath = dataPath+'ghana_imerg_2011_2020_oct.h5'

    a = imergDataset_h5(fPath, start_date, end_date, 12, 12)
    a.__getitem__(0)

    print('stop for debugging')




    


        
