import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import glob
import datetime
import h5py
import json 


import numpy as np
import pandas as pd
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly

from torch.utils.data import Dataset



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
            date_str = file.split("/")[-1].split('.')[1]
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

def create_sample_datasets(dataPath, EF5_events, train_st_dts, train_len, prediction_steps):

    # # if train_st_inds is scalar, then make it a size 1 list
    # if isinstance(train_st_inds, int):
    #     train_st_inds = [train_st_inds]

    in_event_samples, out_event_samples, meta_samples = [], [], []

    for event_ind, event_name in enumerate(EF5_events):

        precipitations, datetimes = load_EF5_data(os.path.join(dataPath, event_name))


        for train_st_dt in train_st_dts[event_ind]:

            train_st_ind = list(datetimes).index(train_st_dt)
            # create one sample of "complete" data
            train_ed_ind = train_st_ind + train_len
            training_ind = np.arange(train_st_ind, train_ed_ind)
            pred_ind = np.arange(train_ed_ind, train_ed_ind+prediction_steps)   

            # inputs
            in_event_samples.append(precipitations[:, :, training_ind])

            # observed outputs
            out_event_samples.append(precipitations[:, :, pred_ind])

            # metadata
            in_datatimes_str = [datetimes[ind].strftime('%Y-%m-%d %H:%M:%S') for ind in training_ind]    
            out_datatimes_str = [datetimes[ind].strftime('%Y-%m-%d %H:%M:%S') for ind in pred_ind]    
            
            meta_samples.append(pd.Series({'event_name': event_name,\
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
        print(idx)
        # To load dataset
        with h5py.File(self.fPath,'r') as hf:
        # hf = h5py.File(self.fPath,'r')
            X = hf['IN_Precipitations'][idx, :, :, :]
            Y = hf['OUT_Precipitations'][idx, :, :, :]

        X_dt = self.meta.iloc[idx]['in_datetimes']
        # X_dt_str = self.meta.iloc[idx]['in_datetimes'].split(',')
        # X_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in X_dt_str]

        Y_dt = self.meta.iloc[idx]['out_datetimes'] 
        # Y_dt_str = self.meta.iloc[idx]['out_datetimes'].split(',')  
        # Y_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in Y_dt_str] 

        event_name = self.meta.iloc[idx]['event_name']
            
        return (X, Y, X_dt, Y_dt, event_name)
    


#===============================================================================
#===========================Load GeoTiff format data============================
# =============================================================================== 
def get_EF5_geotiff_metadata(fPath='/home/cc/projects/nowcasting/data/EF5/imerg_giotiff_meta.json'):

    with open(fPath, "r") as outfile:
      meta = json.load(outfile)
    
    nx = meta['nx']
    ny = meta['ny'] 
    gt = meta['gt'] 
    proj = meta['proj'] 

    return nx, ny, gt, proj

def WriteGrid(gridOutName, dataOut, nx, ny, gt, proj):
    #Writes out a GeoTIFF based on georeference information in RefInfo
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(gridOutName, nx, ny, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dataOut.shape = (-1, nx)
    dst_ds.GetRasterBand(1).WriteArray(dataOut, 0, 0)
    dst_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    dst_ds = None


def write_forcasts_to_geotiff(output_fPath, output_meta_fPath, resultsPath, model_config):
    nx, ny, gt, proj = get_EF5_geotiff_metadata()

    output_meta = pd.read_csv(output_meta_fPath)    
    with h5py.File(output_fPath,'r') as hf:
        output = hf['forcasts'][:]


    for i in range(output.shape[0]):
        i_meta = output_meta.iloc[i]
        # event_name = i_meta['event_name']
        out_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in i_meta['out_dt'].split(',')]

        method_path = os.path.join(resultsPath, f"{model_config['method']}")
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        

        sample_results_path = os.path.join(method_path, str(i))
        if not os.path.exists(sample_results_path):
            os.mkdir(sample_results_path)
        
        precipitations = output[i, :, :, :] 
        for t in range(precipitations.shape[2]):
            precip_t = precipitations[:, :, t]
            gridOutName = os.path.join(sample_results_path, f"{out_dt[t].strftime('%Y%m%d%H%M')}.tif")
            WriteGrid(gridOutName, precip_t, nx, ny, gt, proj)




#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    


    EF5_events = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Ghana _07_03_2023', 'Nigeria_18_06_2020']
    dataPath = "/home/cc/projects/nowcasting/data/EF5"





    


        
