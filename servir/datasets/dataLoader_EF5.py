import os

import h5py



import numpy as np
import pandas as pd


from torch.utils.data import Dataset
from servir.utils.tiff_images_utils import tiff2h5py    



def create_sample_datasets(dataPath, EF5_events, train_st_dts, train_len, prediction_steps):

    # # if train_st_inds is scalar, then make it a size 1 list
    # if isinstance(train_st_inds, int):
    #     train_st_inds = [train_st_inds]

    in_event_samples, out_event_samples, meta_samples = [], [], []

    for event_ind, event_name in enumerate(EF5_events):

        precipitations, datetimes = tiff2h5py(os.path.join(dataPath, event_name))


        for train_st_dt in train_st_dts[event_ind]:

            train_st_ind = list(datetimes).index(train_st_dt)
            # create one sample of "complete" data
            train_ed_ind = train_st_ind + train_len
            training_ind = np.arange(train_st_ind, train_ed_ind)
            pred_ind = np.arange(train_ed_ind, train_ed_ind+prediction_steps)   

            # inputs
            in_event_samples.append(precipitations[:, :, training_ind])

            # observed outputs
            out_event_samples.append(precipitations[:, :, pred_ind])

            # metadata
            in_datatimes_str = [datetimes[ind].strftime('%Y-%m-%d %H:%M:%S') for ind in training_ind]    
            out_datatimes_str = [datetimes[ind].strftime('%Y-%m-%d %H:%M:%S') for ind in pred_ind]    
            
            meta_samples.append(pd.Series({'event_name': event_name,\
                                            'in_datetimes' : ','.join(in_datatimes_str), \
                                            'out_datetimes' : ','.join(out_datatimes_str) }))
            

    in_event_samples = np.array(in_event_samples)
    out_event_samples= np.array(out_event_samples)  




class EF5Dataset(Dataset):
    def __init__(self, fPath, metaPath):
        self.fPath = fPath
        # To load dataset
        with h5py.File(self.fPath,'r') as hf:
            self.Xall = hf['IN_Precipitations'][:, :, :, :]
            self.Yall = hf['OUT_Precipitations'][:, :, :, :]

        # To load meta data
        self.meta = pd.read_csv(metaPath, index_col=0)

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):

        X = self.Xall[idx, :, :, :]
        Y = self.Yall[idx, :, :, :]

        X_dt = self.meta.iloc[idx]['in_datetimes']
        # X_dt_str = self.meta.iloc[idx]['in_datetimes'].split(',')
        # X_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in X_dt_str]

        Y_dt = self.meta.iloc[idx]['out_datetimes'] 
        # Y_dt_str = self.meta.iloc[idx]['out_datetimes'].split(',')  
        # Y_dt = [datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in Y_dt_str] 

        event_name = self.meta.iloc[idx]['event_name']
            
        return (X, Y, X_dt, Y_dt, event_name)
    





#===================================================================================================
#===================================================================================================
#===================================================================================================
if __name__=='__main__':
    


    EF5_events = ["Côte d'Ivoire_18_06_2018", "Cote d'Ivoire_25_06_2020", 'Ghana _10_10_2020', 'Ghana _07_03_2023', 'Nigeria_18_06_2020']
    dataPath = "/home/cc/projects/nowcasting/data/EF5"





    


        
