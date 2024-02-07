
import sys
base_path ='/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"
sys.path.append(base_path)
import datetime
import h5py
import numpy as np

from torch.utils.data import Dataset
from servir.utils.tiff_images_utils import tiff2h5py  



# load data from h5 file
def load_wa_imerg_data_from_h5(fPath, start_date, end_date):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """
    precipitation = []
    times = []

    with h5py.File(fPath, 'r') as hf:
        precipitation = hf['precipitations'][:]
        times = hf['timestamps'][:]
        times = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in times])
        mean = hf['mean'][()]
        std = hf['std'][()]  

    st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    ind = (times>=st_dt) & (times<end_dt)

    requested_precipitation = precipitation[ind]
    requested_times = times[ind]

    return requested_precipitation, requested_times, mean, std

class waImergDataset(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, max_rainfall_intensity=60.0, normalize=False):

        self.precipitations, self.datetimes, self.mean, self.std = load_wa_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        

        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize:
            self.precipitations = (self.precipitations - self.mean)/self.std
        else:
            self.precipitations =  self.precipitations / max_rainfall_intensity

        # cut off 2 columns of data
        self.precipitations = self.precipitations[:, :, 1:-1]

        print('stop for debugging')

    def __len__(self):
        return self.precipitations.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)


        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = self.precipitations[in_ind]
        out_imgs = self.precipitations[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        return X, Y


class waImergDataset_withMeta(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, 
                max_rainfall_intensity = 60.0, normalize=False):

        self.precipitations, self.datetimes, self.mean, self.std = load_wa_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)

    
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize:
            self.precipitations = (self.precipitations - self.mean)/self.std
        else:
            self.precipitations =  self.precipitations / max_rainfall_intensity

        # cut off 2 columns of data
        self.precipitations = self.precipitations[:, :, 1:-1]


    def __len__(self):
        return self.precipitations.shape[0]-self.in_seq_length-self.out_seq_length

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        in_ind = range(idx, idx+self.in_seq_length)
        out_ind = range(idx+self.in_seq_length, idx+self.in_seq_length+self.out_seq_length)


        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = self.precipitations[in_ind]
        out_imgs = self.precipitations[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        # metadata for a sample
        X_dt = self.datetimes[in_ind]
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in X_dt] 
        X_dt_str = ','.join(X_str)


        Y_dt = self.datetimes[out_ind]
        Y_dt_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in Y_dt]
        Y_dt_str = ','.join(Y_dt_str)

        return (X, Y, X_dt_str, Y_dt_str)



#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    

    dataPath = "/home/cc/projects/nowcasting/data/wa_imerg/"

    # start_date = '2020-07-01'
    # end_date = '2020-08-01' 
    # fPath = dataPath+'wa_imerg.h5'

    # a = waImergDataset(fPath, start_date, end_date, 12, 12)
    # a.__getitem__(0)

    # print('stop for debugging')




    


        
