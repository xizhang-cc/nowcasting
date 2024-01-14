import os
import h5py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



def load_mit_servir_data(data_path, TRAIN_VAL_FRAC=0.8, N_TRAIN=-1, N_TEST=-1):

    # Target locations of sample training & testing data
    DEST_TRAIN_FILE= os.path.join(data_path,'nowcast_training_000.h5')
    DEST_TRAIN_META=os.path.join(data_path, 'nowcast_training_000_META.csv')
    DEST_TEST_FILE= os.path.join(data_path, 'nowcast_testing_000.h5')
    DEST_TEST_META= os.path.join(data_path, 'nowcast_testing_000_META.csv')


    # shape`[N,L,L,T]`
    # `N` is the batch size/sample size
    # `L` is the size of the image patch
    # `T` is the number of time frames in the video (1 time step = 5 minutes).  


    # Loading data takes a few minutes
    with h5py.File(DEST_TRAIN_FILE,'r') as hf:
        Nr = N_TRAIN if N_TRAIN>=0 else hf['IN_vil'].shape[0]
        X_train = hf['IN_vil'][:Nr]
        Y_train = hf['OUT_vil'][:Nr]
        training_meta = pd.read_csv(DEST_TRAIN_META).iloc[:Nr]
        X_train,X_val=np.split(X_train,[int(TRAIN_VAL_FRAC*Nr)])
        Y_train,Y_val=np.split(Y_train,[int(TRAIN_VAL_FRAC*Nr)])
        training_meta,val_meta=np.split(training_meta,[int(TRAIN_VAL_FRAC*Nr)])
            
    with h5py.File(DEST_TEST_FILE,'r') as hf:
        Nr = N_TEST if N_TEST>=0 else hf['IN_vil'].shape[0]
        X_test = hf['IN_vil'][:Nr]
        Y_test = hf['OUT_vil'][:Nr]
        testing_meta=pd.read_csv(DEST_TEST_META).iloc[:Nr]


    return X_train, Y_train, X_val, Y_val, X_test, Y_test, training_meta, val_meta, testing_meta



class ServirDataset(Dataset):
    def __init__(self, X, Y):
        super(ServirDataset, self).__init__()

        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        img_in = self.X[index]
        img_out = self.Y[index]

        # reshape to [T, C, H, W]
        # T: time steps
        # C: channels, 1 if grayscale, 3 if RGB
        # H: height
        # W: width  
        img_in = np.transpose(np.expand_dims(img_in, axis=0), (3, 0, 1, 2)) 
        img_out = np.transpose(np.expand_dims(img_out, axis=0), (3, 0, 1, 2))

        return img_in, img_out