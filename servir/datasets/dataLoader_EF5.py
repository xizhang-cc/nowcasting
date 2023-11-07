import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import glob
import datetime
import h5py


import numpy as np
import pandas as pd
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly
import torch
from torch.utils.data import Dataset


from servir.utils import processIMERG

def load_EF5_data(fPath):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (not sorted by time)
        times (np.array): np.array of date times that match 1:q with precipitation
    """
    precipitation = []
    times = []
    files = glob.glob(fPath+'/*.tif')
    if len(files)>0:
        for file in files:
            tiff_data = gdal.Open(file, GA_ReadOnly)
            imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())
            date_str = file.split('/')[-1].split('.')[1]
            year = date_str[0:4]
            month = date_str[4:6]
            day = date_str[6:8]
            hour = date_str[8:10]
            minute = date_str[10:12]
            dt = datetime.datetime.strptime(year + '-'+ month + '-' + day + ' '+ hour + ':' + minute, '%Y-%m-%d %H:%M')
            times.append(dt)
            precipitation.append(imageArray)

        times = np.array(times)
        precipitation = np.dstack(precipitation)

        sorted_index_array = np.argsort(times)
        sorted_timestamps = times[sorted_index_array]
        sorted_precipitation = precipitation[:, :, sorted_index_array]

    else:
        sorted_precipitation = None
        sorted_timestamps = None

    return sorted_precipitation, sorted_timestamps

def save2h5py_with_metadata():


    with h5py.File(os.path.join(dataPath,'EF5.h5py'),'w') as hf:
        precipitations = []
        meta_df = pd.DataFrame()
        for ind, event_name in enumerate(EF5_events):

            precipitation, datetimes = load_EF5_data(os.path.join(dataPath, event_name, 'processed_imerg'))
        
            if precipitation is not None:

                datetimes_str = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetimes]

                meta_df = pd.concat([
                                     meta_df, \
                                     pd.DataFrame({'event_name':event_name, 'datetimes':','.join(datetimes_str) }, index=[ind]) \
                                     ])
                
                precipitations.append(precipitation)

        
        dset = hf.create_dataset('precipitations', data=np.array(precipitations))
        meta_df.to_csv(os.path.join(dataPath, 'EF5_meta.csv'))  
    

def create_sample_datasets(dataPath, train_st_inds, train_len, prediction_steps):
    # To load meta data
    meta = pd.read_csv(os.path.join(dataPath, 'EF5_meta.csv'), index_col=0)
    meta['datetimes'] = meta['datetimes'].apply(lambda x: [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')\
                                                           for dt_str in x.split(',')])

    # To load dataset
    with h5py.File(os.path.join(dataPath,'EF5.h5py'),'r') as hf:
        data = hf['precipitations'][:]

    in_event_samples, out_event_samples, meta_samples = [], [], []

    for event_idx in range(data.shape[0]):

        event_data = data[event_idx, :, :, :]
        event_meta = meta.iloc[event_idx]

        for train_st_ind in train_st_inds:

            # create one sample of "complete" data
            train_ed_ind = train_st_ind + train_len
            training_ind = np.arange(train_st_ind, train_ed_ind)
            pred_ind = np.arange(train_ed_ind, train_ed_ind+prediction_steps)   

            # inputs
            in_event_samples.append(event_data[:, :, training_ind])

            # in_meta_samples.append(pd.Series({'event_name':event_meta['event_name'], 'datetimes':','.join(in_datatimes_str) }))

            # observed outputs
            out_event_samples.append(event_data[:, :, pred_ind])

            # metadata
            in_datatimes_str = [event_meta['datetimes'][ind].strftime('%Y-%m-%d %H:%M:%S') for ind in training_ind]    
            out_datatimes_str = [event_meta['datetimes'][ind].strftime('%Y-%m-%d %H:%M:%S') for ind in pred_ind]    
            
            meta_samples.append(pd.Series({'event_name':event_meta['event_name'],\
                                            'in_datetimes' : ','.join(in_datatimes_str), \
                                            'out_datetimes' : ','.join(out_datatimes_str) }))

    in_event_samples = np.array(in_event_samples)
    out_event_samples= np.array(out_event_samples)  

    meta_samples = pd.DataFrame(meta_samples)   

    with h5py.File(os.path.join(dataPath,'EF5_samples.h5py'),'w') as hf:
        din = hf.create_dataset('IN_Precipitations', data=in_event_samples)
        dout = hf.create_dataset('OUT_Precipitations', data=out_event_samples)  

    meta_samples.to_csv(os.path.join(dataPath, 'EF5_samples_meta.csv'))



class EF5Dataset(Dataset):
    def __init__(self, fPath, metaPath):
        self.fPath = fPath
        # To load meta data
        self.meta = pd.read_csv(metaPath, index_col=0)  

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        # To load dataset
        with h5py.File(self.fPath,'r') as hf:
        # hf = h5py.File(self.fPath,'r')
            X = hf['IN_Precipitations'][idx, :, :, :]
            Y = hf['OUT_Precipitations'][idx, :, :, :]
        
        X_dt_str = self.meta.iloc[idx]['in_datetimes'].split(',')
        X_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in X_dt_str]

        Y_dt_str = self.meta.iloc[idx]['out_datetimes'].split(',')  
        Y_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in Y_dt_str] 
            
        return X, Y, X_dt, Y_dt
    

def get_EF5_geotiff_metadata(dataPath):
    xmin = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1

    # choose a random raw event to get geo metadata 
    # '3B-HHR-E.MS.MRG.3IMERG.20180618-S123000-E125959.0750.V06B.30min'
    f_str = os.path.join(dataPath, "Côte d'Ivoire_18_06_2018/raw_imerg/3B-HHR-E.MS.MRG.3IMERG.20180618-S000000-E002959.0000.V06B.30min.tif")
    # f_str = f'data/{event_name}/processed_imerg/imerg.{dt.strftime("%Y%m%d%H%M")}.30minAccum.tif'
    _, nx, ny, gt, proj = processIMERG(f_str,xmin,ymin,xmax,ymax)

    return nx, ny, gt, proj

#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    
    EF5_events = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Ghana _07_03_2023', 'Nigeria_18_06_2020']
    dataPath = "/home/cc/projects/nowcasting/data/EF5"

    train_st_inds = np.arange(0, 8)
    train_len = 10
    prediction_steps = 8

    # create_sample_datasets(dataPath, train_st_inds, train_len, prediction_steps)

    ef5 = EF5Dataset(os.path.join(dataPath,'EF5_samples.h5py'), os.path.join(dataPath, 'EF5_samples_meta.csv'))

    X, Y, X_dt, Y_dt = ef5.__getitem__(0)

    print('stop for debugging') 
    


        
