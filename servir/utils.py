import os
import h5py
import numpy as np
import datetime


def create_folder(path, level = 3):
    """Create a folder if it does not exist level by level.

    Args:
        path (str): path to the folder
        level (int): level of the folder to create
    """
    paths = path.rsplit('/', level)
    for i in range(1,level+1,1):
        if not os.path.exists(os.path.join(*paths[:i+1])):
            os.makedirs(path)


# load data from h5 file
def load_imerg_data_from_h5(fPath, start_date=None, end_date=None):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """


    with h5py.File(fPath, 'r') as hf:
        precipitation = hf['precipitations'][:]
        times = hf['timestamps'][:]
        times = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in times])
        mean = hf['mean'][()]   
        std = hf['std'][()]
        max = hf['max'][()]
        min = hf['min'][()]

    if (start_date is not None) and (end_date is not None):
        st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        ind = (times>=st_dt) & (times<=end_dt)

        precipitation = precipitation[ind]
        times = times[ind]

    return precipitation, times, mean, std, max, min


def load_IR_data_from_h5(fPath, start_date=None, end_date=None):
    """Function to load IMERG tiff data from the associate event folder

    Args:
        data_location (str): string path to the location of the event data

    Returns:
        precipitation (np.array): np.array of precipitations (sorted by time)
        times (np.array): np.array of date times
    """

    with h5py.File(fPath, 'r') as hf:
        imgs = hf['IRs'][:]
        times = hf['timestamps'][:]
        times = np.array([datetime.datetime.strptime(x.decode('utf-8'), '%Y-%m-%d %H:%M:%S') for x in times])
        mean = hf['mean'][()]
        std = hf['std'][()]
        max = hf['max'][()]
        min = hf['min'][()]
        

    if (start_date is not None) and (end_date is not None):
   
        st_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        ind = (times>=st_dt) & (times<=end_dt)

        imgs = imgs[ind]
        times = times[ind]


    return imgs, times, mean, std, max, min