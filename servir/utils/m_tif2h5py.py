
import os
import sys
import glob
import datetime
import h5py

import numpy as np
import osgeo.gdal as gdal
from osgeo.gdalconst import GA_ReadOnly


####

# This file is for project pipline only! 
# The function below is used to convert all tiff images in a folder to h5py 
# file with filename 'imerg_{start_date}_{end_date}.h5'

####
"""Function to load IMERG tiff data from the associate event folder

Args:
    sys.argv[2] (str): string path to the location of the event data

Returns:
    precipitation (np.array): np.array of precipitations (not sorted by time)
    times (np.array): np.array of date times that match 1:q with precipitation

    Save precipitation and times in string format to h5py file
"""
# tif_directory = '/home/cc/projects/nowcasting/temp/'
# h5_directory = '/home/cc/projects/nowcasting/temp/'

tif_directory = sys.argv[1]
h5_directory = sys.argv[2]


filename_extension = 'tif'


if len(sys.argv) != 3:
    print("Usage: scriptname <path_to_hot_folder>")
    exit(1)



if os.path.isdir(tif_directory) is False:
    print("The supplied directory ({}) does not exist.".format(tif_directory))
    exit(1)

files = glob.glob(tif_directory + '/*.' + filename_extension)

if not files:
    print("No files with estension {} found in {}.".format(
        filename_extension, tif_directory))
    exit(1)

precipitation = []
times = []


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
# images in tensor [T, H, W]
precipitation = np.transpose(np.dstack(precipitation), (2, 0, 1))

sorted_index_array = np.argsort(times)
sorted_timestamps = times[sorted_index_array]
sorted_precipitation = precipitation[sorted_index_array]

st_dt = sorted_timestamps[0].strftime('%Y%m%d%H%M')
end_dt = sorted_timestamps[-1].strftime('%Y%m%d%H%M')
fname = f"imerg_{st_dt}_{end_dt}.h5"


sorted_timestamps_dt = [x.strftime('%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]
with h5py.File(os.path.join(h5_directory, fname), 'w') as hf:
    hf.create_dataset('precipitations', data=sorted_precipitation)
    hf.create_dataset('timestamps', data=sorted_timestamps_dt)



    