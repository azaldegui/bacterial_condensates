# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:04:17 2023

@author: azaldec
"""

import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import scipy.stats as ss
import json
from scipy.stats import ttest_ind


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

directory = sys.argv[1]
filenames = filepull(directory)
font_size = 8


ys = ['mCherry 0', 'mCherry 30','DAPI 0', 'DAPI 30' ]
colors = ['darkgreen', 'darkgreen', 'violet', 'violet']
style = ['solid','dashed', 'solid','dashed' ]
fig, axes = plt.subplots(ncols = 2, nrows=2, figsize=(4.25,3.25), 
                         dpi=200)

plt.rcParams.update({'font.size': 8})    
plt.rcParams['font.family'] = 'Arial'
sns.set(font="Arial")
plt.rcParams['svg.fonttype'] = 'none'
ax = axes.ravel()
for ii, file in enumerate(filenames):
    
    df = pd.read_csv(file,index_col=None)
    norm_df = (df-df.min()) / (df.max() - df.min()) 
    
    

    
    for ee, y in enumerate(ys):
        sns.lineplot(ax = ax[ii], data=norm_df, x = 'Position', y=ys[ee],
                     linestyle=style[ee], color=colors[ee])
        ax[ee].set_xlabel('Cell axis', fontsize=font_size)
        ax[ee].set_ylabel('Norm. Fluorescence', fontsize=font_size)
        ax[ee].tick_params(axis='x',labelsize=font_size)
        ax[ee].tick_params(axis='y',labelsize=font_size)
        
fig.tight_layout()   
plt.savefig(directory[:-5] + 'cipro_wt_profiles.svg', dpi=300)  
plt.show()
    
    
    
    