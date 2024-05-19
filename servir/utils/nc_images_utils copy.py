import os
import sys
base_path = '/home/cc/projects/nowcasting' #"/home1/zhang2012/nowcasting/"#
sys.path.append(base_path)
import glob
import datetime
import h5py
import glob

import numpy as np
import netCDF4 as nc
from scipy.interpolate import griddata



# Resample function
def resample_Tb(old_lat, old_lon, old_data, lat_R, lon_R):
    old_coordinates = np.array(np.meshgrid(old_lat, old_lon)).T.reshape(-1, 2)
    new_coordinates = np.array(np.meshgrid(lat_R, lon_R)).T.reshape(-1, 2)

    new_data = np.full((len(lat_R), len(lon_R), old_data.shape[2]), np.nan)
    
    for t in range(old_data.shape[2]):
        new_data[:, :, t] = griddata(old_coordinates, old_data[:, :, t].flatten(), new_coordinates, method='nearest').reshape(len(lat_R), len(lon_R))
        
    return new_data


def nc2h5py(dataPath, start_date, end_date, fname='wa_nc.h5'):
##==================Data Loading=====================##
    # find all .nc files
    files = glob.glob(dataPath+'/raw/*.nc')

    # load the first file to get the lat and lon
    nc_data_t = nc.Dataset(os.path.join(dataPath, files[0]),'r')
    lat_t= nc_data_t.variables['lat'][:]
    lon_t= nc_data_t.variables['lon'][:]
    lat_R= np.arange(-2.9, 33.1, 0.1)
    lon_R = np.arange(-21.4, 30.4, 0.1)


    lambda_val = 10.8
    nu = 10000 / lambda_val
    c1 = 1.19104E-5
    c2 = 1.43877

    # get only the files that are within the date range
    st = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    et = datetime.datetime.strptime(end_date, '%Y-%m-%d') 

    requested_files = []
    test = []
    for file in files:
        f_dt = datetime.datetime.strptime(file.split('HRSEVIRI_')[1][:13], '%Y%m%dT%H%M')

        if f_dt < et and f_dt >= st:
            requested_files.append(file)
            test.append(f_dt)

    test.sort()

    file_size = len(requested_files)
    batch_size = 10
    batches = file_size // batch_size

    # start loading the data
    IR_imgs = [] 
    IR_dts = []
    for i in range(batches):

        print(f'batch: {i}')

        ch9 = np.full((len(lat_t), len(lon_t), batch_size), np.nan)
        Tb = np.full((len(lat_t), len(lon_t), batch_size), np.nan)

        for j in range(batch_size):
            index = i*batch_size+j
            file = requested_files[index]

            nc_data_t = nc.Dataset(file)
            date_str = file.split('HRSEVIRI_')[1][:13]  # Extract date string from file name
            date_obj = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M')

            IR_dts.append(date_obj)

            ch9[:, :, j] = nc_data_t.variables['channel_9'][:]
            Tb[:, :, j] = c2 * nu / np.log(1 + (c1 * nu**3 /ch9[:, :, j]))

        Tb_R= resample_Tb(lat_t, lon_t, Tb, lat_R, lon_R)
        Tb_R = Tb_R.transpose(2, 0, 1)  # [S, H, W]

        IR_imgs.append(Tb_R)

    # get the remaining files
    remaining_files_size = file_size - batches*batch_size
    ch9 = np.full((len(lat_t), len(lon_t), remaining_files_size), np.nan)
    Tb = np.full((len(lat_t), len(lon_t), remaining_files_size), np.nan)

    for j in range(remaining_files_size):
        index = batches* batch_size+j 
        file = requested_files[index]
        nc_data_t = nc.Dataset(file)
        
        # get datetime object
        date_str = file.split('HRSEVIRI_')[1][:13]  # Extract date string from file name
        date_obj = datetime.datetime.strptime(date_str, '%Y%m%dT%H%M')
        IR_dts.append(date_obj)

        ch9[:, :, j] = nc_data_t.variables['channel_9'][:]
        Tb[:, :, j] = c2 * nu / np.log(1 + (c1 * nu**3 /ch9[:, :, j]))

    Tb_R= resample_Tb(lat_t, lon_t, Tb, lat_R, lon_R)
    Tb_R = Tb_R.transpose(2, 0, 1)  # [S, H, W]

    IR_imgs.append(Tb_R)

    #==================================================================================================#

    # convert to numpy array
    IR_imgs = np.concatenate(IR_imgs, axis=0)

    # sort the files by datetime
    IR_dts = np.array(IR_dts)
    sorted_index_array = np.argsort(IR_dts)
    sorted_timestamps = IR_dts[sorted_index_array]

    sorted_IR = IR_imgs[sorted_index_array]

    # find mean and std and save into json file
    # IR_mean = np.mean(requested_IRs)
    # IR_std = np.std(requested_IRs)
    sorted_timestamps_dt = [x.strftime('%Y-%m-%d %H:%M:%S') for x in sorted_timestamps]

    with h5py.File(os.path.join(dataPath, fname), 'w') as hf:
        hf.create_dataset('IRs', data=sorted_IR)
        hf.create_dataset('timestamps', data=sorted_timestamps_dt)
        # hf.create_dataset('mean', data=IR_mean)
        # hf.create_dataset('std', data=IR_std)




if __name__ == "__main__":

    dataPath = os.path.join(base_path, 'data', 'wa_IR')
    start_date = '2011-10-01'
    end_date = '2011-11-01'
    nc2h5py(dataPath, start_date, end_date, fname='wa_IR_08.h5')

    # # print('stop for debugging')

    # with h5py.File(os.path.join(dataPath, 'wa_IR.h5'), 'r') as hf:
    #     imgs = hf['IRs'][:]
    #     img_dts = hf['timestamps'][:]
    #     mean = imgs.mean()
    #     std = imgs.std()
    #     max = imgs.max()
    #     min = imgs.min()

    # print(f'mean = {mean}')
    # print(f'std = {std}')
    # print(f'max = {max}')
    # print(f'min = {min}')

        

    # # flip images up-down
    # imgs = np.dstack([np.flipud(imgs[k]) for k in range(imgs.shape[0])]).transpose(2, 0, 1)

    # # cropping 2 columns from the left and right
    # imgs = imgs[:, :, 1:-1]


    # imgs_IR = np.concatenate([imgs_06, imgs_07, imgs_08], axis=0)
    # print(f'imgs_IR shape: {imgs_IR.shape}')
    # img_IR_dts = img_dts_06 + img_dts_07 + img_dts_08
    # print(f'len(img_IR_dts): {len(img_IR_dts)}')

    # print(f'img_IR_dts[0]: {img_IR_dts[0]}')
    # print(f'img_IR_dts[-1]: {img_IR_dts[-1]}')


    # with h5py.File(os.path.join(dataPath, 'wa_IR.h5'), 'w') as hf:
    #     hf.create_dataset('IRs', data=imgs_IR)
    #     hf.create_dataset('timestamps', data=img_IR_dts)
    #     hf.create_dataset('mean', data = imgs_IR.mean())
    #     hf.create_dataset('std', data = imgs_IR.std())

