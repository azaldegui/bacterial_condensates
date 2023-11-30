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

dfs = []
samples=  ['cI_agg_02', 'cI_agg_04', 'cI_agg_10', 'McdB_01', 'McdB_02', 'McdB_03']
#samples = ['mCherry', 'mCherry-D8N', 'mNeonGreen']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]
for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
 
all_data = pd.concat(dfs)
all_data = all_data.reset_index()
all_data = all_data[all_data['Time'] < 310]
# make a plot for each sample

# normalize to t = 0 
norm_data = []
for sample in samples:
    
    samp_data = all_data[all_data['Sample'] == sample]
    
    samp_data['I = 0.1'] = samp_data['I = 0.1'] / samp_data['I = 0.1'].iloc[0,]
    
    norm_data.append(samp_data)
norm_data = pd.concat(norm_data)
#all_data['I = 0.5'] = all_data['I = 0.5'].multiply(100)


fig, axes = plt.subplots(figsize=(3.25, 2.5), dpi=300)

font_size = 9
    
plt.rcParams.update({'font.size': font_size})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'


 
axia = sns.lineplot( data = norm_data, y="I = 0.1", x="Time", hue="Protein",  #hue_order=samples,
                   errorbar='sd', linewidth=2, marker='o',markersize=5,markeredgecolor='black',
                   palette=['royalblue','purple'],
                   )
handles, labels = axia.get_legend_handles_labels()
axia.legend([handles[0],handles[1]] , ['cI_agg', 'McdB'], loc='lower right', ncol=2, frameon=False)
axes.set_ylim(0, 1.2)
#ax[jj].set_ylim(10**2, 6000)
#ax[jj].set_yscale('log')
axes.set_ylabel('Norm. Condensation Coefficient', 
                       fontsize=font_size)
axes.set_xlabel('Time (s)', fontsize=font_size)
axes.tick_params(axis='x',labelsize=font_size)
axes.tick_params(axis='y',labelsize=font_size)   
    
   
    
    
    
   
    
fig.tight_layout()   
plt.savefig(directory[:-5] + 'lysisplot.svg', dpi=300) 
plt.show()