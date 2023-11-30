# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:37:42 2022

@author: azaldegc
"""

import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


# user defined parameters
font_size = 12
framerate = .5 # seconds

# read csv files
# covert file to dataframe
directory = sys.argv[1]
filenames = filepull(directory)
dfs = []
filenames.sort()
df = pd.read_csv(filenames[0])   
print(df)
# order to plot the samples
samples_order = ['cI', 'Pop6', 
                 'Pop78', 'McdB']

# initiate figure
fig, axes = plt.subplots(ncols=2,nrows=len(samples_order), figsize=(8,12), 
                                 sharey=True, sharex=True)
ax = axes.ravel()

# for each protein
for nn, sample in enumerate(samples_order):
    
    # use data for protein
    data = df[df['Sample']==sample]
    
    frap_curves = []
    
    # for each FRAP experiment
    for ii in range(len(data)):
        
        frap_curve = data.iloc[ii].to_numpy()[2:] # frap curve
        frap_curves.append(frap_curve)
        time = [x*framerate for x in range(len(frap_curve))] # time series
        ax[nn*2].plot(time, frap_curve, '-', c='k', alpha=0.7) # plot frap curve
    
    # take avg and stdev of the frap curves
    frap_curves_arr = np.array(frap_curves)
    avg_frap = []
    std_frap = []
    for jj in range(len(time)):
        avg_frap.append(np.mean(frap_curves_arr[:, jj]))
        std_frap.append(np.std(frap_curves_arr[:, jj]))
    avg_frap = np.asarray(avg_frap)
    std_frap = np.asarray(std_frap)
    
    
    # plot the avg and stdev of the frap curves
    ax[nn*2+1].plot(time, avg_frap, '--', c='r', alpha=1, linewidth=3)
    ax[nn*2+1].fill_between(time, avg_frap - std_frap,
                       avg_frap + std_frap, alpha=0.5, color='r')
    ax[nn*2].set_ylim(-.1,)
    ax[nn*2+1].set_ylim(-.1,1.1)
    ax[nn*2].set_xlim(0,)
    ax[nn*2+1].set_xlim(0,)

    ax[nn*2].set_ylabel('Norm. fluor.', fontsize=font_size)
    ax[nn*2+1].legend([sample])    
    
ax[nn*2].set_xlabel('Time (s)', fontsize=font_size)
ax[nn*2+1].set_xlabel('Time (s)', fontsize=font_size)   
fig.tight_layout()
plt.savefig(directory[:-15] + "all_frap.png", dpi=300) # save figure
plt.show() # show figure