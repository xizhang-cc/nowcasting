# to download IMERG early run data from online
# Import all libraries
import sys
import subprocess
import os
import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly
#from gdalconst import GA_ReadOnly
#from gdalconst import *
import numpy as np


from pysteps.io.nowcast_importers import import_netcdf_pysteps
from pysteps.datasets import  create_default_pystepsrc
import netCDF4 as nc
import time
import glob
import numpy as np
import netCDF4 as nc

import datetime as DT
import osgeo.gdal as gdal
from osgeo.gdal import gdalconst
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
    for file in files:
        tiff_data = gdal.Open(file, GA_ReadOnly)
        imageArray = np.array(tiff_data.GetRasterBand(1).ReadAsArray())
        date_str = file.split('/')[-1].split('.')[1]
        year = date_str[0:4]
        month = date_str[4:6]
        day = date_str[6:8]
        hour = date_str[8:10]
        minute = date_str[10:12]
        dt = DT.datetime.strptime(year + '-'+ month + '-' + day + ' '+ hour + ':' + minute, '%Y-%m-%d %H:%M')
        times.append(dt)
        precipitation.append(imageArray)
    precipitation = np.array(precipitation)

    return precipitation, times

def sort_IMERG_data(precipitation, times):
    """Function to sort the imerg data based on precitpitation and times array

    Args:
        precipitation (np.array): numpy array of precipitation images
        times (np.array): numpy array of datetime objects that match 1:1 with precipitation array

    Returns:
        sorted_precipitation (np.array): sorted numpy array of precipitation images
        sorted_timestamps (np.array): sorted numpy array of datetime objects that match 1:1 with sorted precipitation array

    """
    sorted_index_array = np.argsort(times)
    # print(sorted_index_array)
    sorted_timestamps = np.array(times)[sorted_index_array]
    sorted_precipitation = np.array(precipitation)[sorted_index_array]

    timestep = np.diff(sorted_timestamps)

    # Let's inspect the shape of the imported data array
    print("Shape of the sorted precipitation array", sorted_precipitation.shape)

    return sorted_precipitation, sorted_timestamps
