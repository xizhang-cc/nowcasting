import os

from pysteps import verification


from servir.methods.extrapolation_methods import lp_nowcast, linda_nowcast, steps_nowcast
from servir.methods.naive_persistence import naive_persistence

training_config = {
    'batch_size': 1,
    'metrics': ['fss'],
    'dataset': 'EP5'
    }


methods_dict = {
                'STEPS': {'func': steps_nowcast, 'kargs': {'n_ens_members': 20, 'n_cascade_levels': 6}}, \
                'LINDA': {'func': linda_nowcast, 'kargs': {'max_num_features': 15, 'add_perturbations': False}}, \
                'Lagrangian_Persistence': {'func': lp_nowcast, 'kargs': {}},\
                }



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


batch_size,batch_num=1,0
# FSS score
# calculate FSS
fss = verification.get_method("FSS")

thr=1.0
scale=2


# load data
init_IMERG_config_pysteps()



# observed precipitation .gif creation
path_outputs = 'results/'+event_name

if not os.path.isdir(path_outputs):
    os.mkdir(path_outputs)


    
    for method in methods_dict.keys():
        paras = methods_dict[method]
        pfunc = paras['func'] 
        kargs = paras['kargs']
        #==========Forcast===========

        forcast_precip = pfunc(train_precip, setup['prediction_steps'], **kargs)

            
    










