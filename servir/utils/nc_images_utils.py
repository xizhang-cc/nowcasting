import os
import sys
base_path ='/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"
sys.path.append(base_path)
import glob
import datetime
import h5py
import glob

import numpy as np
import netCDF4 as nc

method_name = 'ConvLSTM'
dataset_name = 'wa_imerg'


##==================Data Loading=====================##
# where to load data
dataPath = os.path.join(base_path, 'data', 'IR' , 'july2020_raw/')


dir_t = [filename for filename in os.listdir(dataPath ) if filename.startswith('HRSEVIRI_20200701T') and filename.endswith('.nc')]
filename_t = dataPath + dir_t[0]
nc_data_t = nc.Dataset(filename_t,'r')
# variable_t=nc_data_t.variables
dir_t.sort()  # sorted chronologically
lat_t= nc_data_t.variables['lat'][:]
lon_t= nc_data_t.variables['lon'][:]


for d in np.arange(1, 32):
    pass
   
dates_t = []   
ch9= np.full((len(lat_t), len(lon_t), len(dir_t)), np.nan)
Tb= np.full((len(lat_t), len(lon_t), len(dir_t)), np.nan)


ch9= nc_data_t.variables['channel_9']


ch9 = np.full((len(lat_t), len(lon_t), len(dir_t)), np.nan)
Tb = np.full((len(lat_t), len(lon_t), len(dir_t)), np.nan)


lambda_val = 10.8
nu = 10000 / lambda_val
c1 = 1.19104E-5
c2 = 1.43877


for i in range(len(dir_t)):
    filename_t= dataPath+dir_t[i]
    nc_data_t = nc.Dataset(filename_t)
    date_str = filename_t.split('HRSEVIRI_')[1][:13]  # Extract date string from file name
    date_obj = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M')
    dates_t.append(date_obj)
    ch9[:, :, i] = nc_data_t.variables['channel_9'][:]
    Tb[:, :, i] = c2 * nu / np.log(1 + (c1 * nu**3 /ch9[:, :, i]))

# Resample function

from scipy.interpolate import griddata


def resample_Tb(old_lat, old_lon, old_data, lat_R, lon_R):
    old_coordinates = np.array(np.meshgrid(old_lat, old_lon)).T.reshape(-1, 2)
    new_coordinates = np.array(np.meshgrid(lat_R, lon_R)).T.reshape(-1, 2)

    new_data = np.full((len(lat_R), len(lon_R), old_data.shape[2]), np.nan)
    
    for t in range(old_data.shape[2]):
        new_data[:, :, t] = griddata(old_coordinates, old_data[:, :, t].flatten(), new_coordinates, method='nearest').reshape(len(lat_R), len(lon_R))
        
    return new_data



lat_R= np.arange(-2.9, 33.1, 0.1)
lon_R = np.arange(-21.4, 30.4, 0.1)

Tb_R= resample_Tb(lat_t, lon_t, Tb, lat_R, lon_R)

Tb_R = Tb_R.transpose(2, 0, 1)  # [S, H, W]







# dir_t = glob.glob(dataPath+'/*.nc')

# filename_t = dir_t[0]

# nc_data_t = nc.Dataset(filename_t,'r')
# variable_t=nc_data_t.variables

# ch9= nc_data_t.variables['channel_9']


print('stop for debugging')


