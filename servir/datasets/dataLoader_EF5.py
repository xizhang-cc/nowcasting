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
    # if train_st_inds is scalar, then make it a size 1 list
    if isinstance(train_st_inds, int):
        train_st_inds = [train_st_inds]


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
def ReadandWarp(gridFile,xmin,ymin,xmax,ymax):
    #Read grid and warp to domain grid
    #Assumes no reprojection is necessary, and EPSG:4326
    rawGridIn = gdal.Open(gridFile, GA_ReadOnly)
    # Adjust grid
    pre_ds = gdal.Translate('OutTemp.tif', rawGridIn, options="-co COMPRESS=Deflate -a_nodata 29999 -a_ullr -180.0 90.0 180.0 -90.0")

    gt = pre_ds.GetGeoTransform()
    proj = pre_ds.GetProjection()
    nx = pre_ds.GetRasterBand(1).XSize
    ny = pre_ds.GetRasterBand(1).YSize
    NoData = 29999
    pixel_size = gt[1]

    #Warp to model resolution and domain extents
    ds = gdal.Warp('', pre_ds, srcNodata=NoData, srcSRS='EPSG:4326', dstSRS='EPSG:4326', dstNodata='-9999', format='VRT', xRes=pixel_size, yRes=-pixel_size, outputBounds=(xmin,ymin,xmax,ymax))

    WarpedGrid = ds.ReadAsArray()
    new_gt = ds.GetGeoTransform()
    new_proj = ds.GetProjection()
    new_nx = ds.GetRasterBand(1).XSize
    new_ny = ds.GetRasterBand(1).YSize

    return WarpedGrid, new_nx, new_ny, new_gt, new_proj

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

def processIMERG(local_filename,llx,lly,urx,ury):
  # Process grid
  # Read and subset grid
  NewGrid, nx, ny, gt, proj = ReadandWarp(local_filename,llx,lly,urx,ury)

  # Scale value
  NewGrid = NewGrid*0.1

  return NewGrid, nx, ny, gt, proj


def get_EF5_geotiff_metadata():
    xmin = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1

    # choose a random raw event to get geo metadata 
    # '3B-HHR-E.MS.MRG.3IMERG.20180618-S123000-E125959.0750.V06B.30min'
    dataPath = "/home/cc/projects/nowcasting/data/EF5"
    f_str = os.path.join(dataPath, "Côte d'Ivoire_18_06_2018/raw_imerg/3B-HHR-E.MS.MRG.3IMERG.20180618-S000000-E002959.0000.V06B.30min.tif")
    # f_str = f'data/{event_name}/processed_imerg/imerg.{dt.strftime("%Y%m%d%H%M")}.30minAccum.tif'
    _, nx, ny, gt, proj = processIMERG(f_str,xmin,ymin,xmax,ymax)

    return nx, ny, gt, proj

def write_forcasts_to_geotiff(output_fPath, output_meta_fPath, resultsPath, model_config):
    nx, ny, gt, proj = get_EF5_geotiff_metadata()

    output_meta = pd.read_csv(output_meta_fPath)    
    with h5py.File(output_fPath,'r') as hf:
        output = hf['forcasts'][:]


    for i in range(output.shape[0]):
        i_meta = output_meta.iloc[i]
        event_name = i_meta['event_name']
        out_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in i_meta['out_dt'].split(',')]


        event_results_path = os.path.join(resultsPath, event_name)
        if not os.path.exists(event_results_path):
            os.mkdir(os.path.join(resultsPath, event_name))
        
        precipitations = output[i, :, :, :] 
        for t in range(precipitations.shape[2]):
            precip_t = precipitations[:, :, t]
            gridOutName = os.path.join(event_results_path, f"{model_config['method']}_forcast_{out_dt[t].strftime('%Y%m%d%H%M')}.30minAccum.tif")
            WriteGrid(gridOutName, precip_t, nx, ny, gt, proj)




#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    


    EF5_events = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Ghana _07_03_2023', 'Nigeria_18_06_2020']
    dataPath = "/home/cc/projects/nowcasting/data/EF5"


    save2h5py_with_metadata()
    train_st_inds = 0
    train_len = 10
    prediction_steps = 8

    create_sample_datasets(dataPath, train_st_inds, train_len, prediction_steps)



    


        
