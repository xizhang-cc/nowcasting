import os
import sys
sys.path.append("/home/cc/projects/nowcasting/")
import h5py

import numpy as np
import pandas as pd

def extrapolation_model_setup(model_config):

    if model_config['method'] == 'LINDA':
        from servir.methods.extrapolation_methods import linda
        pfunc = linda
        kargs = {'max_num_features': model_config['max_num_features'], 'add_perturbations': model_config['add_perturbations']}

    elif model_config['method'] == 'STEPS':
        from servir.methods.extrapolation_methods import steps
        pfunc = steps
        kargs = {'n_ens_members': model_config['n_ens_members'], 'n_cascade_levels': model_config['n_cascade_levels']}

    return pfunc, kargs



def forcasts_and_save(dataloader, model_config, output_fPath, output_meta_fPath, save=True):
    ## forcasts and save results
    pfunc, kargs = extrapolation_model_setup(model_config)

    forcasts = []
    forcasts_meta = []  
    for in_imgs, out_imgs, in_dt, out_dt, event_name  in dataloader:

        in_imgs = in_imgs.numpy()       
        in_imgs = np.squeeze(in_imgs, axis=0)
        in_imgs = np.transpose(in_imgs, (2, 0, 1))

        out_imgs = out_imgs.numpy()
        out_imgs = np.squeeze(out_imgs, axis=0)
        out_imgs = np.transpose(out_imgs, (2, 0, 1))

        forcast_precip = pfunc(in_imgs, out_imgs.shape[0], **kargs)

        forcast_precip = np.transpose(forcast_precip, (1, 2, 0))

        forcasts.append(forcast_precip) 
        forcasts_meta.append(pd.Series({'event_name':event_name[0], 'in_dt':in_dt[0], 'out_dt':out_dt[0]}))

        print("stop for debugging")

    forcasts = np.array(forcasts)
    forcasts_meta = pd.DataFrame(forcasts_meta) 

    if save==True:
        with h5py.File(output_fPath,'w') as hf:
            preds = hf.create_dataset('forcasts', data=forcasts)

        forcasts_meta.to_csv(output_meta_fPath)

    return forcasts, forcasts_meta

