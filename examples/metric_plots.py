import numpy as np

# import csv file from lightning_logs
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

# load the csv file
csv_file = '/home/cc/projects/nowcasting/lightning_logs/version_26/metrics.csv'

df = pd.read_csv(csv_file)

# plot the train/g_loss 
fig = plt.figure(figsize=(10, 5))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

# ax.plot(df['epoch'], df['train/g_loss'], label='train/g_loss')
x = df['train/d_loss']
x = x[~np.isnan(x)]
ax.plot(x)