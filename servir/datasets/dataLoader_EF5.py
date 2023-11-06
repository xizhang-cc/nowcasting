import glob
import datetime


import numpy as np
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly



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

def get_EF5_geotiff_metadata(dataPath):
    # choose a random raw event to get geo metadata 
    # '3B-HHR-E.MS.MRG.3IMERG.20180618-S123000-E125959.0750.V06B.30min'
    f_str = os.path.join(dataPath, "Côte d'Ivoire_18_06_2018/raw_imerg/3B-HHR-E.MS.MRG.3IMERG.20180618-S000000-E002959.0000.V06B.30min.tif")
    # f_str = f'data/{event_name}/processed_imerg/imerg.{dt.strftime("%Y%m%d%H%M")}.30minAccum.tif'
    _, nx, ny, gt, proj = processIMERG(f_str,xmin,ymin,xmax,ymax)

    return nx, ny, gt, proj

#===================================================================================================
#==================The main return h5py files from original processed IMERG data====================
#===================================================================================================
if __name__=='__main__':
    import os
    import sys
    sys.path.append("/home/cc/projects/nowcasting/")

    import h5py
    import pandas as pd
    from servir.utils import processIMERG

    xmin = -21.4
    xmax = 30.4
    ymin = -2.9
    ymax = 33.1

    EF5_events = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Ghana _07_03_2023', 'Nigeria_18_06_2020']
    dataPath = "/home/cc/projects/nowcasting/data/EF5"

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
            
    
    ## To load dataset
    # with h5py.File(os.path.join(dataPath,'EF5.h5py'),'r') as hf:
    #     data = hf['precipitations']

    ## To lad meta data
    # meta = pd.read_csv(os.path.join(dataPath, 'EF5_meta.csv'), index_col=0)

