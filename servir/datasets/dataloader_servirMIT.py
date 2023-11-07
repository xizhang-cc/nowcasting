import os
import h5py

import numpy as np
import pandas as pd

data_path="/home/cc/projects/nowcasting/data/mit_servir"
# Target locations of sample training & testing data
DEST_TRAIN_FILE= os.path.join(data_path,'nowcast_training_000.h5')
DEST_TRAIN_META=os.path.join(data_path, 'nowcast_training_000_META.csv')
DEST_TEST_FILE= os.path.join(data_path, 'nowcast_testing_000.h5')
DEST_TEST_META= os.path.join(data_path, 'nowcast_testing_000_META.csv')


# shape`[N,L,L,T]`
# `N` is the batch size/sample size
# `L` is the size of the image patch
# `T` is the number of time frames in the video (1 time step = 5 minutes).  
# set how much use for validation
TRAIN_VAL_FRAC=0.8

N_TRAIN=100
N_TEST=100
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

batch_size,batch_num=8,0
bs,be=batch_size*batch_num,batch_size*(batch_num+1)
X,Y,meta = X_train[bs:be],Y_train[bs:be],training_meta.iloc[bs:be]

print('stop for debugging')


import torch
from torch.utils.data import Dataset


class ServirDataset(Dataset):
    def __init__(self, X, Y, normalize=False):
        super(ServirDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None

        if normalize:
            # get the mean/std values along the channel dimension
            mean = data.mean(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            std = data.std(axis=(0, 1, 2, 3)).reshape(1, 1, -1, 1, 1)
            data = (data - mean) / std
            self.mean = mean
            self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels