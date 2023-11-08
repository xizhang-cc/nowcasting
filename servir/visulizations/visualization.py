import os
import glob
import shutil

import imageio
from pysteps.visualization.animations import animate

metadata = {'accutime': 30.0,
    'cartesian_unit': 'degrees',
    'institution': 'NOAA National Severe Storms Laboratory',
    'projection': '+proj=longlat  +ellps=IAU76',
    'threshold': 0.0125,
    'timestamps': None,
    'transform': None,
    'unit': 'mm/h',
    'x1': -21.4,
    'x2': 30.4,
    'xpixelsize': 0.04,
    'y1': -2.9,
    'y2': 33.1,
    'yorigin': 'upper',
    'ypixelsize': 0.04,
    'zerovalue': 0}



def create_precipitation_gif(precipitations,timestamps_obs, timestep_min, geodata, path_outputs, title, gif_dur = 1000):



    temp_path = f"{path_outputs}/temp"

    if not os.path.isdir(temp_path):
        os.mkdir(temp_path)


    animate(precipitations, timestamps_obs  = timestamps_obs,
            timestep_min = timestep_min, geodata=geodata, title=title, \
            savefig=True, fig_dpi=100, fig_format='png', path_outputs=temp_path)
    

    # load images to create .gif file
    images = []
    forcast_precip_imgs = sorted(glob.glob(f"{temp_path}/*.png") )
    for img in forcast_precip_imgs:
        images.append(imageio.imread(img))

    kargs = { 'duration': gif_dur }
    imageio.mimsave(f"{path_outputs}/{title}.gif", images, **kargs)

    # remove temp folder
    shutil.rmtree(temp_path)


