import os
from datetime import datetime, timedelta
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly


class imergDataset_tif(Dataset):
    def __init__(self, fPath, start_datetime, end_datetime, in_seq_length, out_seq_length, sampling_freq = timedelta(hours=2), normalize_method='01range', \
                 **kwargs):
        
        self.fPath = fPath
        # convert str to datetime
        self.start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        self.end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    
    
        self.time_delta = timedelta(minutes=30) # this is fixed for IMERG data
        self.sampling_freq = sampling_freq # sliding window sampling frequency
        self.normalize_method = normalize_method

        if normalize_method == 'gaussian':
            assert 'precip_mean' in kwargs.keys(), 'mean value is required for gaussian normalization'
            assert 'precip_std' in kwargs.keys(), 'std value is required for gaussian normalization'
            self.mean = kwargs['precip_mean']
            self.std = kwargs['precip_std']
        elif normalize_method == '01range':
            assert 'precip_max' in kwargs.keys(), 'max value is required for 01range normalization'
            assert 'precip_min' in kwargs.keys(), 'min value is required for 01range normalization'
            self.max = kwargs['precip_max']
            self.min = kwargs['precip_min']    

        if 'img_shape' in kwargs.keys():
            img_shape = kwargs['img_shape']

            self.img_height = img_shape[0]
            self.img_width = img_shape[1]
        else: # original IMERG data shape
            self.img_height = 360
            self.img_width = 518


    def __len__(self):

        num_images = int((self.end_datetime - self.start_datetime) / self.time_delta)

        return (num_images - self.in_seq_length - self.out_seq_length + 1) // (self.sampling_freq // self.time_delta)

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        index = idx*(self.sampling_freq // self.time_delta)

        item_st = self.start_datetime+ index*self.time_delta
        # get all datetime from item_st to item_ed every time_delta
        item_dts = [item_st + k*self.time_delta for k in range(self.in_seq_length + self.out_seq_length)]
        # get all file paths
        item_files = [os.path.join(self.fPath, f'imerg.{x.strftime("%Y%m%d%H%M")}.30minAccum.tif') for x in item_dts]

        precipitations = []
        for file in item_files:
            if not os.path.exists(file):
                print(f'file {file} does not exist')
                # use the last image if the file does not exist
                imageArray = precipitations[-1]
            else:
                tiff_data = gdal.Open(file, GA_ReadOnly)
                imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())

            precipitations.append(imageArray)

        # desired shape: [T, C, H, W]
        # concatenate the images along the time axis
        precipitations = np.stack(precipitations, axis=0)
        # add channel dimension, resulting [B, C, H, W]
        precipitations = np.expand_dims(precipitations, axis=(1))

        # crop the image to the desired shape(centor crop)
        if self.img_height != 360:
            h_start = (360 - self.img_height) // 2
            precipitations = precipitations[:, :, h_start:h_start+self.img_height, :]

        if self.img_width != 518:
            w_start = (518 - self.img_width) // 2
            precipitations = precipitations[:, :, :, w_start:w_start+self.img_width]


        # normalize the data
        if self.normalize_method == 'gaussian':
            precipitations = (precipitations - self.mean) / self.std
        elif self.normalize_method == '01range':
            precipitations =  (precipitations - self.min) / (self.max - self.min)   

        # input and output images for a sample
        in_imgs = precipitations[:self.in_seq_length]   
        out_imgs = precipitations[self.in_seq_length:self.in_seq_length+self.out_seq_length]    


        return in_imgs, out_imgs


class imergDataset_tif_withMeta(Dataset):
    def __init__(self, fPath, start_datetime, end_datetime, in_seq_length, out_seq_length, sampling_freq = timedelta(hours=2), normalize_method='01range', \
                 **kwargs):
        
        self.fPath = fPath
        # convert str to datetime
        self.start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M:%S')
        self.end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M:%S')
        self.in_seq_length = in_seq_length
        self.out_seq_length = out_seq_length    
    
        self.time_delta = timedelta(minutes=30) # this is fixed for IMERG data
        self.sampling_freq = sampling_freq # sliding window sampling frequency
        self.normalize_method = normalize_method

        if normalize_method == 'gaussian':
            assert 'precip_mean' in kwargs.keys(), 'mean value is required for gaussian normalization'
            assert 'precip_std' in kwargs.keys(), 'std value is required for gaussian normalization'
            self.mean = kwargs['precip_mean']
            self.std = kwargs['precip_std']
        elif normalize_method == '01range':
            assert 'precip_max' in kwargs.keys(), 'max value is required for 01range normalization'
            assert 'precip_min' in kwargs.keys(), 'min value is required for 01range normalization'
            self.max = kwargs['precip_max']
            self.min = kwargs['precip_min']    


    def __len__(self):

        num_images = int((self.end_datetime - self.start_datetime) / self.time_delta)

        return (num_images - self.in_seq_length - self.out_seq_length + 1) // (self.sampling_freq // self.time_delta)

    def __getitem__(self, idx):
        # desire to [T, C, H, W]
            
            # T: time steps
            # C: channels, 1 if grayscale, 3 if RGB
            # H: height
            # W: width 

        index = idx*(self.sampling_freq // self.time_delta)

        item_st = self.start_datetime+ index*self.time_delta
        # get all datetime from item_st to item_ed every time_delta
        item_dts = [item_st + k*self.time_delta for k in range(self.in_seq_length + self.out_seq_length)]
        # get all file paths
        item_files = [os.path.join(self.fPath, f'imerg.{x.strftime("%Y%m%d%H%M")}.30minAccum.tif') for x in item_dts]

        precipitations = []
        for file in item_files:
            if not os.path.exists(file):
                print(f'file {file} does not exist')
                # use the last image if the file does not exist
                imageArray = precipitations[-1]
            else:
                tiff_data = gdal.Open(file, GA_ReadOnly)
                imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())

            precipitations.append(imageArray)

        # desired shape: [T, C, H, W]
        # concatenate the images along the time axis
        precipitations = np.stack(precipitations, axis=0)
        # add channel dimension
        precipitations = np.expand_dims(precipitations, axis=(1))
    

        # normalize the data
        if self.normalize_method == 'gaussian':
            precipitations = (precipitations - self.mean) / self.std
        elif self.normalize_method == '01range':
            precipitations =  (precipitations - self.min) / (self.max - self.min)   

        # input and output images for a sample
        in_imgs = precipitations[:self.in_seq_length]   
        out_imgs = precipitations[self.in_seq_length:self.in_seq_length+self.out_seq_length]    

        # metadata for a sample
        item_dts_str = [x.strftime('%Y-%m-%d %H:%M:%S') for x in item_dts]
        in_dt_str = ','.join(item_dts_str[:self.in_seq_length])
        out_dt_str = ','.join(item_dts_str[self.in_seq_length:self.in_seq_length+self.out_seq_length])


        return (in_imgs, out_imgs, in_dt_str, out_dt_str)


class WAImergDataModule(LightningDataModule):
   

    def __init__(
        self,
        dataPath: str = "/home/cc/projects/nowcasting/data/wa_imerg/",
        train_start_date: str = '2019-01-01 00:00:00',
        train_end_date: str = '2020-07-31 23:30:00',
        val_start_date: str = '2020-08-01 00:00:00',
        val_end_date: str = '2020-09-30 23:30:00',
        test_start_date: str = '2020-10-01 00:00:00',
        test_end_date: str = '2020-10-31 23:30:00',
        in_seq_length: int = 4,
        out_seq_length: int = 12,
        sampling_freq: timedelta = timedelta(hours=2),
        normalize_method: str = '01range',
        precip_mean: float = 0.0,
        precip_std: float = 1.0,
        precip_max: float = 1.0,
        precip_min: float = 0.0,
        img_shape: tuple = (360, 518)
    ):
        super().__init__()

        self.imergTrain = imergDataset_tif(dataPath, train_start_date, train_end_date, in_seq_length, out_seq_length,\
                                        sampling_freq=sampling_freq, normalize_method=normalize_method, img_shape = img_shape,\
                                        precip_mean=precip_mean, precip_std=precip_std, precip_max=precip_max, precip_min=precip_min)
        self.imergVal = imergDataset_tif(dataPath, val_start_date, val_end_date, in_seq_length, out_seq_length,\
                                        sampling_freq=sampling_freq, normalize_method=normalize_method,img_shape = img_shape,\
                                        precip_mean=precip_mean, precip_std=precip_std, precip_max=precip_max, precip_min=precip_min)
        self.imergTest = imergDataset_tif(dataPath, test_start_date, test_end_date, in_seq_length, out_seq_length,\
                                        sampling_freq=sampling_freq, normalize_method=normalize_method,img_shape = img_shape,\
                                        precip_mean=precip_mean, precip_std=precip_std, precip_max=precip_max, precip_min=precip_min)


    def train_dataloader(self):
        return DataLoader(self.imergTrain, batch_size=2, pin_memory=True, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.imergVal, batch_size=2, pin_memory=True, shuffle=False, num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.imergTest, batch_size=2, pin_memory=True, shuffle=False,num_workers=4)
    

#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    

    dataPath = "/home/cc/projects/nowcasting/data/wa_imerg/"



    a = imergDataset_tif(dataPath, '2019-03-01 00:00:00', '2019-03-31 23:30:00',\
                                   4, 12,normalize_method=None)
    l = a.__len__()
    a.__getitem__(368)

    print('stop for debugging')




    


        
