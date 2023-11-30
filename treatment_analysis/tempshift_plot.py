# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:13:44 2023

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


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)
font_size = 15
dfs = []
samples=  ['cI_agg', 'PopTag_SL', 'PopTag_LL', 'McdB']
#samples = ['mCherry', 'mCherry-D8N', 'mNeonGreen']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]
for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
 
all_data = pd.concat(dfs)
all_data = all_data.reset_index()
all_data = all_data[all_data['Time (min)'] < 32]
all_data['I = 0.5'] = all_data['I = 0.5'].multiply(100)
# make a plot for each sample




fig, axes = plt.subplots(figsize=(4., 3), dpi=300)


    
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'


font_size = 9
 
axia = sns.lineplot( data = all_data, y="I = 0.5", x="Time (min)", hue="Sample", hue_order=samples,
                   errorbar=('ci', 95), linewidth=2, marker='o',markersize=5,markeredgecolor='black',
                   palette=colors,
                   )
handles, labels = axia.get_legend_handles_labels()
axia.legend(handles , samples, loc='upper right', ncol=2, frameon=False)
axes.set_ylim(10, 100)
#axes.set_xlim(-.1, 17)
#ax[jj].set_ylim(10**2, 6000)
#ax[jj].set_yscale('log')
axes.set_ylabel('Condensation Coefficient', 
                       fontsize=font_size)
axes.set_xlabel('Time (min)', fontsize=font_size)
axes.tick_params(axis='x',labelsize=font_size)
axes.tick_params(axis='y',labelsize=font_size)   
    
   
    
    
    
   
    
fig.tight_layout()   
plt.savefig(directory[:-5] + 'tempshift_plot.svg', dpi=300) 
plt.show()