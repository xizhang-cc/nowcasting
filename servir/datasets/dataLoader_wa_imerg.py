import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import glob
import datetime
import h5py
import json 


import numpy as np
import pandas as pd
# import osgeo.gdal as gdal
# from osgeo.gdalconst import GA_ReadOnly

from torch.utils.data import Dataset


# 'imerg.2020 0701 0000.30minAccum.tiff'
# 'imerg.2020 0731 2330.30minAccum.tiff' 


# def load_wa_imerg_data(fPath, start_date, end_date, save2h5=False, fname='raw_images.h5'):
#     """Function to load IMERG tiff data from the associate event folder

#     Args:
#         data_location (str): string path to the location of the event data

#     Returns:
#         precipitation (np.array): np.array of precipitations (not sorted by time)
#         times (np.array): np.array of date times that match 1:q with precipitation
#     """
#     precipitation = []
#     times = []
#     files = glob.glob(os.path.join(fPath, 'july2020_raw/imerg.2020*.tif'))

#     if len(files)>0:
#         for file in files:
#             tiff_data = gdal.Open(file, GA_ReadOnly)
#             imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())
#             date_str = file.split("/")[-1].split('.')[1]
#             year = date_str[0:4]
#             month = date_str[4:6]
#             day = date_str[6:8]
#             hour = date_str[8:10]
#             minute = date_str[10:12]
#             dt = datetime.datetime.strptime(year + '-'+ month + '-' + day + ' '+ hour + ':' + minute, '%Y-%m-%d %H:%M')

#             if dt >= datetime.datetime.strptime(start_date, '%Y-%m-%d') and dt < datetime.datetime.strptime(end_date, '%Y-%m-%d'):
#                 times.append(dt)
#                 precipitation.append(imageArray)

#         times = np.array(times)
#         # images in tensor [T, H, W]
#         precipitation = np.transpose(np.dstack(precipitation), (2, 0, 1))

#         sorted_index_array = np.argsort(times)
#         sorted_timestamps = times[sorted_index_array]
#         sorted_precipitation = precipitation[sorted_index_array]

#     else:
#         sorted_precipitation = None
#         sorted_timestamps = None

#     if save2h5: 
#         sorted_timestamps_dt = [x.strftime('%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]
#         with h5py.File(os.path.join(fPath, fname), 'w') as hf:
#             hf.create_dataset('precipitations', data=sorted_precipitation)
#             hf.create_dataset('timestamps', data=sorted_timestamps_dt)

#     return sorted_precipitation, sorted_timestamps


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

    st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    ind = (times>=st_dt) & (times<end_dt)

    requested_precipitation = precipitation[ind]
    requested_times = times[ind]

    return requested_precipitation, requested_times

class waImergDataset(Dataset):
    def __init__(self, fPath, start_date, end_date, in_seq_length, out_seq_length, 
                max_rainfall_intensity = 52, normalize=False):

        self.precipitations, self.datetimes = load_wa_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)
        
        # pixel value range (0, 1)
        self.precipitations =  self.precipitations / max_rainfall_intensity
    
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        if normalize:
            self.mean = np.mean(self.precipitations, axis=(0, 1, 2))
            self.std = np.std(self.precipitations, axis=(0, 1, 2))
            self.precipitations = (self.precipitations - self.mean)/self.std



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
                max_rainfall_intensity = 52, normalize=False):

        self.precipitations, self.datetimes = load_wa_imerg_data_from_h5(fPath, start_date= start_date, end_date=end_date)

        # pixel value range (0, 1)
        self.precipitations =  self.precipitations / max_rainfall_intensity
    
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    

        if normalize:
            self.mean = np.mean(self.precipitations, axis=(0, 1, 2))
            self.std = np.std(self.precipitations, axis=(0, 1, 2))
            self.precipitations = (self.precipitations - self.mean)/self.std


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



    load_wa_imerg_data(dataPath, start_date='2020-07-01', end_date='2020-08-01',\
                        save2h5=True, fname='imerg_2020_july.h5py')

    # precipitation, timestamps = create_sample_datasets(dataPath)
    # start_date = '2020-07-01'
    # end_date = '2020-08-01' 
    # fPath = dataPath+f'/imerg_{start_date}_{end_date}.h5'

    # load_wa_imerg_data_from_h5(fPath, start_date='2020-07-01', end_date='2020-07-08')


    print('stop for debugging')




    


        
