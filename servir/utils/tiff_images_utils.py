
import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import glob
import datetime
import h5py


import numpy as np
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly



def tiff2h5py(fPath, start_date, end_date, save2h5=False, fname='raw_images.h5'):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (not sorted by time)
        times (np.array): np.array of date times that match 1:q with precipitation
    """
    precipitation = []
    times = []
    files = glob.glob(os.path.join(fPath, 'july2020_raw/imerg.2020*.tif'))

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

            if dt >= datetime.datetime.strptime(start_date, '%Y-%m-%d') and dt < datetime.datetime.strptime(end_date, '%Y-%m-%d'):
                times.append(dt)
                precipitation.append(imageArray)

        times = np.array(times)
        # images in tensor [T, H, W]
        precipitation = np.transpose(np.dstack(precipitation), (2, 0, 1))

        sorted_index_array = np.argsort(times)
        sorted_timestamps = times[sorted_index_array]
        sorted_precipitation = precipitation[sorted_index_array]

    else:
        sorted_precipitation = None
        sorted_timestamps = None

    if save2h5: 
        sorted_timestamps_dt = [x.strftime('%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]
        with h5py.File(os.path.join(fPath, fname), 'w') as hf:
            hf.create_dataset('precipitations', data=sorted_precipitation)
            hf.create_dataset('timestamps', data=sorted_timestamps_dt)

    return sorted_precipitation, sorted_timestamps



# ===============================================================================
# ===========================Load GeoTiff format data============================
# =============================================================================== 
import json
def get_EF5_geotiff_metadata(fPath='/home/cc/projects/nowcasting/data/wa_imerg/imerg_giotiff_meta.json'):

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
        out_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in i_meta['out_datetimes'].split(',')]

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