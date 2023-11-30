# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:49:16 2022

@author: azaldegc
"""


import seaborn as sns
import sys
import numpy as np
import pandas as pd
import glob
from PIL import Image
import tifffile as tif
import matplotlib.pyplot as plt
from scipy import stats as stat
import statistics as ss
import math as math
from skimage import (feature, filters, measure)



# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]  
    
    return filenames



directory = sys.argv[1]
print(directory)
filenames = filepull(directory)
print(filenames)
dfs = []
samples= []
for file in filenames:
    
    dff = pd.read_csv(file,index_col=None )
    dfs.append(dff)
    samples.append(dff['Sample'].iloc[0])
    
colors = ['gray','darkred']    
all_data = pd.concat(dfs, ignore_index=True)
samples = list(set(samples))
samples.sort()
print(all_data)
var_plot = 'uM'
df = all_data.astype({var_plot: float, 'BlobBool': float})
order_ = ['PopTag_LL', 'McdB']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.25,3), 
                               sharey=True, sharex=False, dpi=300)

ax = axes.ravel()
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'

font_size = 8

#axel = axes.ravel()
for ii,sample in enumerate(samples):
    protein_df = all_data[all_data['Sample']==sample]
    axviol = sns.violinplot(ax=ax[ii],x='Sample', y='uM', data=all_data,split=True,
                    zorder=0, cut=0, bw='scott', hue='BlobBool', 
                    linewidth=2, scale='width', scale_hue=True,
                     inner=None, palette={0: "white", 1: "white"})

    for nn, item in enumerate(axviol.collections):
        if nn == 0 or nn == 2:
            item.set_edgecolor('gray')
        #x0, y0, width, height = item.get_paths()[0].get_extents().bounds
      #  item.set_clip_path(plt.Rectangle((x0-0.01, y0), (width)/2, height,
       #                               transform=axviol.transData))
      
    
    if nn == 1 or nn == 3:
         item.set_edgecolor('darkred')
      #  x0, y0, width, height = item.get_paths()[0].get_extents().bounds
       
      #item.set_clip_path(plt.Rectangle((x0-0.01, y0), (-1*width)/2, height,
       #                                 transform=axviol.transData))
    
    num_items = len(axviol.collections)
    
    g = sns.stripplot(ax=ax[ii], x='Sample', y='uM', dodge=True,
                  data = protein_df, size=4, zorder=1, edgecolor='black', linewidth=0,
                  alpha=0.2, jitter=0.3, hue='Experiment_no',order = order_,
                  palette=colors)
    
    g.set_xticklabels(['No focus', 'Focus'])   


       
'''      
sns.kdeplot(ax=axel[0], data=all_data, x="uM",palette=colors,
            hue="BlobBool", multiple='stack', common_norm=False,)
           # kde=True, stat='density',binwidth=10, )

sns.histplot(ax=axel[1], data=all_data, x="uM", stat='density', 
            hue="BlobBool", multiple='stack', bins = 20,
            element="step", fill=False,
            cumulative=True, common_norm=True)
'''
#sns.ecdfplot(data=all_data, x='uM', hue='BlobBool', palette=colors, linewidth=3)

#axel[1].get_legend().set_title("Focus detected")
for ii in range(2):
   # axel[ii].set_xlabel(r'Concentration$_{\rm app}$ ($\mu$M)', 
   #                   fontsize=font_size)
    ax[ii].legend([],[], loc='upper right',frameon=False )
    ax[ii].set_ylim(-20,300)
    ax[ii].set_ylabel(r'Concentration$_{\rm app}$' + u' (\u03bcM)',fontsize=font_size)
    ax[ii].set_xlabel('',fontsize=font_size)
   # axel[1].set_ylabel('Fraction',fontsize=font_size) 
   # axel[1].set_xlabel(r'Concentration$_{\rm app}$' + u' (\u03bcM)',fontsize=font_size)
   
    ax[ii].tick_params(axis='x',labelsize=font_size)
    ax[ii].tick_params(axis='y',labelsize=font_size)
fig.tight_layout()
plt.savefig(directory[:-5] + 'copycounting.svg', dpi=300)
plt.show()

for sample in samples:
    print(sample)
    protein_df = all_data[all_data['Sample']==sample]
    for exp in range(3):
        print(exp)
        exp_df = protein_df[protein_df['Experiment_no']==exp+1]
        print("classfied", exp_df.groupby('BlobBool')[var_plot].describe())
