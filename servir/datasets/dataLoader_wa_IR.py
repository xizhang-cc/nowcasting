import os
import sys
# base_path ='/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"
# sys.path.append(base_path)
import datetime
import h5py
import numpy as np

from torch.utils.data import Dataset



# load data from h5 file
def load_IR_data_from_h5(dataPath, start_date, end_date):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """

    # with h5py.File(os.path.join(dataPath, 'wa_IR_06.h5'), 'r') as hf:
    #     imgs_06 = hf['IRs'][:]
    #     img_dts = hf['timestamps'][:]
    #     img_dts_06 = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in img_dts]

    # with h5py.File(os.path.join(dataPath, 'wa_IR_07.h5'), 'r') as hf:
    #     imgs_07 = hf['IRs'][:]
    #     img_dts = hf['timestamps'][:]
    #     img_dts_07 = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in img_dts]

    with h5py.File(os.path.join(dataPath, 'wa_IR_08.h5'), 'r') as hf:
        imgs_08 = hf['IRs'][:]
        # temperatry fix for nan to ~mean value
        imgs_08[np.isnan(imgs_08)] = 284

        img_dts = hf['timestamps'][:]
        img_dts_08 = [datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in img_dts]

    # times = np.array( img_dts_06+ img_dts_07 + img_dts_08)
    # imgs = np.concatenate([img_06, imgs_07, imgs_08], axis=0)

    times = np.array(img_dts_08)
    imgs = imgs_08

    st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    ind = (times>=st_dt) & (times<end_dt)
    requested_times = times[ind]
    requested_imgs = imgs[ind]

    return requested_imgs, requested_times



class IRDataset(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, max_temp_in_kelvin=337 ,normalize=False):

        self.imgs, self.datetimes = load_IR_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize:
            pass
            # self.imgs = (self.imgs - self.mean)/self.std
        else:
            self.imgs =  self.imgs / max_temp_in_kelvin

        # crop images
        self.imgs = self.imgs[:, :, 1:-1]

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
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, max_temp_in_kelvin=337 ,normalize=False):

        self.imgs, self.datetimes = load_IR_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        # normalize the data
        if normalize:
            pass
            # self.imgs = (self.imgs - self.mean)/self.std
        else:
            self.imgs =  self.imgs / max_temp_in_kelvin

        # crop images
        self.imgs = self.imgs[:, :, 1:-1]

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
    dataPath = "/home/cc/projects/nowcasting/data/wa_IR/"

    # a = IRDataset(os.path.join(dataPath, fname), '2020-07-01', '2020-07-04', 9, 9)
    # b = a.__getitem__(0)
    # precipitation, timestamps = create_sample_datasets(dataPath)
    # start_date = '2020-06-01'
    # end_date = '2020-09-01' 
    # fPath = dataPath+f'/imerg_{start_date}_{end_date}.h5'

    start_date = '2020-08-01'
    end_date = '2020-08-18'
    a = IRDataset_withMeta(dataPath, start_date, end_date, 10, 9)
    for k in range(10):
        a.__getitem__(k)

    print('stop for debugging')




    


        
