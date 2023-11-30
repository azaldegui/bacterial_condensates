# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:00:47 2022

@author: azaldegc


Script to make parition coefficient plots for Hoang and Azaldegui, et al. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import scipy.stats as ss


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)
font_size = 12
dfs = []
samples= []

for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
    samples.append(df['Label'].iloc[0])
    
all_data = pd.concat(dfs)
all_data = all_data.reset_index()
samples = list(set(samples))
samples.sort()
order_ = ['cI_agg', 'PopTag_SL', 'PopTag_LL', 'McdB', 'McdB_PS-']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'magenta']
print(samples)


fig, axes = plt.subplots(figsize=(3,3),sharey=True, dpi = 250)

PC_data_focus_filtered = all_data.copy()

# plot the rain cloud plot
PC_data_focus_filtered['PC_avg'] = PC_data_focus_filtered[PC_data_focus_filtered['Focus'] != 0]['PC_avg'].apply(lambda x: float(x))
print(PC_data_focus_filtered)
PC_data_focus_filtered = PC_data_focus_filtered[PC_data_focus_filtered['PC_avg'] > 0]

all_partitions = []
for sample in samples:
    
    partitions = PC_data_focus_filtered[PC_data_focus_filtered['Label']==sample]['PC_avg'].to_numpy()
    
    print(sample,'n:', len(partitions) )
    print("parition ratio: ", np.mean(partitions), np.std(partitions), np.std(partitions) / (len(partitions))**0.5)

    for ii in range(len(all_partitions)):
    
        print(ss.ttest_ind(partitions, all_partitions[ii], equal_var=False))
    all_partitions.append(partitions)
axviol = sns.violinplot(ax=axes, y='PC_avg', x='Label', data=PC_data_focus_filtered,
                    color='white', order=order_, zorder=0,
                    linewidth=2, edgecolor='black', scale='width', scale_hue=True,
                    cut=0, inner=None)
    
for item in axviol.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0-0.05, y0), width/2, height,
                       transform=axviol.transData))

num_items = len(axviol.collections)
    
sns.stripplot(ax=axes, x='Label', y='PC_avg', 
              data = PC_data_focus_filtered, size=4, zorder=1, edgecolor='black', linewidth=1,
              alpha=1, jitter=True, order=order_,
              palette=colors)
    
    
# Shift each strip plot strictly below the correponding volin.
for item in axviol.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.1)   


axes.set_ylim(*10**-1,5*10**2)
axes.set_yscale('log')
axes.set_ylabel('Partition ratio', 
                          fontsize=font_size)
axes.set_xlabel('')
axes.tick_params(axis='x',labelsize=font_size)
axes.tick_params(axis='y',labelsize=font_size)

fig.tight_layout()
#plt.savefig(directory[:-5] + '20221024_Figure_partitio_ratio_nomCherry.png', dpi=300)
plt.show()
