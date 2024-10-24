import os
import sys
# base_path ='/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"
# sys.path.append(base_path)
import datetime
import h5py
import numpy as np

from torch.utils.data import Dataset



# load data from h5 file
def load_IR_data_from_h5(fPath, start_date=None, end_date=None):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """

    with h5py.File(fPath, 'r') as hf:
        imgs = hf['IRs'][:]
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

        imgs = imgs[ind]
        times = times[ind]


    return imgs, times, mean, std, max, min

def IR_neg_scale(data, replacevalue=-2.0, threshold=240):
    temp = np.where(data<=threshold, data, np.nan)
    f_max = np.nanmax(temp)
    f_min = np.nanmin(temp)

    new = np.where(data<=threshold, -(2*(data-f_min)/(f_max-f_min)-1), replacevalue)
    return new

class IRDataset(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length , normalize_method='gaussian'):

        self.imgs, self.datetimes, self.mean, self.std, self.max, self.min = load_IR_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize_method=='gaussian':
            self.imgs = (self.imgs - self.mean)/self.std
        elif normalize_method=='01range':
            self.imgs =  1 -  (self.imgs - self.min) / (self.max - self.min)
        elif normalize_method=='thresholded_scale':
            self.imgs = IR_neg_scale(self.IRs)

        all_str_ind = self.datetimes[: -(self.in_seq_length+self.out_seq_length)]
        ind = [x.minute in [0, 30] for x in all_str_ind]
        self.samples = all_str_ind[ind]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        st_dt = self.samples[idx]
        st_ind = list(self.datetimes).index(st_dt)
        
        in_ind = range(st_ind, st_ind+self.in_seq_length)
        out_ind = range(st_ind+self.in_seq_length, st_ind+self.in_seq_length+self.out_seq_length)

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = self.imgs[in_ind]
        out_imgs = self.imgs[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        return X, Y


class IRDataset_withMeta(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length , normalize_method='gaussian'):

        self.imgs, self.datetimes, self.mean, self.std, self.max, self.min = load_IR_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize_method=='gaussian':
            self.imgs = (self.imgs - self.mean)/self.std
        elif normalize_method=='01range':
            self.imgs =  1 -  (self.imgs - self.min) / (self.max - self.min)
        elif normalize_method=='thresholded_scale':
            self.imgs = IR_neg_scale(self.IRs)

        all_str_ind = self.datetimes[: -(self.in_seq_length+self.out_seq_length)]
        ind = [x.minute in [0, 30] for x in all_str_ind]
        self.samples = all_str_ind[ind]

    def __len__(self):
    
        return len(self.samples)

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        st_dt = self.samples[idx]
        st_ind = list(self.datetimes).index(st_dt)
        
        in_ind = range(st_ind, st_ind+self.in_seq_length)
        out_ind = range(st_ind+self.in_seq_length, st_ind+self.in_seq_length+self.out_seq_length)

        # input and output images for a sample
        # current shape: [T, H, W]
        in_imgs = self.imgs[in_ind]
        out_imgs = self.imgs[out_ind]

        # desired shape: [T, C, H, W]
        X = np.expand_dims(in_imgs, axis=(1))
        Y = np.expand_dims(out_imgs, axis=(1))

        # metadata for a sample
        X_dt = self.datetimes[in_ind]
        X_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in X_dt] 
        X_dt_str = ','.join(X_str)


        Y_dt = self.datetimes[out_ind]
        Y_str = [y.strftime('%Y-%m-%d %H:%M:%S') for y in Y_dt]
        Y_dt_str = ','.join(Y_str)

        return (X, Y, X_dt_str, Y_dt_str)
    





#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    
    import os
    # dataPath = "/home/cc/projects/nowcasting/data/wa_IR/"

    # a = IRDataset(os.path.join(dataPath, fname), '2020-07-01', '2020-07-04', 9, 9)
    # b = a.__getitem__(0)
    # precipitation, timestamps = create_sample_datasets(dataPath)
    # start_date = '2020-06-01'
    # end_date = '2020-09-01' 
    # fPath = dataPath+f'/imerg_{start_date}_{end_date}.h5'

    start_date = '2020-06-01'
    end_date = '2020-08-18'
    fPath = '/home1/zhang2012/nowcasting/data/wa_IR/wa_IR.h5'
    a = IRDataset_withMeta(fPath, start_date, end_date, 10, 9)
    for k in range(10):
        X, Y, X_dt_str, Y_dt_str = a.__getitem__(k)
        print(Y_dt_str)

  




    


        
