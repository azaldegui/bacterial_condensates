# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:48:56 2022

@author: azaldegc
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
from sklearn.mixture import GaussianMixture
from scipy import stats as stat

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)
font_size = 15
dfs = []
samples= []
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple']
for file in filenames:
    
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
    samples.append(df['Sample'].iloc[0])
    
all_data = pd.concat(dfs)
samples = list(set(samples))
samples.sort()
order_ = ['cI', 'PopTag-SL', 'PopTag-LL', 'McdB' ]

print(samples)

fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(6, 3), 
                               sharey=False, sharex=True, dpi = 300)
ax = axes.ravel()

sample_lifetimes = []
print(samples)
for sample in samples:
    lifetimes = all_data[all_data['Sample']==sample]['Track Lifetime (hrs)'].to_numpy()
    decay = all_data[all_data['Sample']==sample]['Fluorescence_max_diff'].to_numpy()
    sample_lifetimes.append(lifetimes)
    print(sample,'n:', len(lifetimes) )
    print("Lifetime: ", np.mean(lifetimes), np.std(lifetimes), np.std(lifetimes) / (len(lifetimes))**0.5)
    print("delta fluor: ", np.mean(decay), np.std(decay))
t_stat, p = stat.ttest_ind(sample_lifetimes[2],sample_lifetimes[3], equal_var=False)
print(f't={t_stat}, p={p}')
ii = 0

#################################################################
axviol = sns.violinplot(ax=ax[ii], x='Track Lifetime (hrs)', y='Sample', data=all_data,
                    color='white', order=order_, zorder=0,
                    linewidth=2, edgecolor='black', scale='width', scale_hue=True,
                    cut=0, inner=None)

for item in axviol.collections:
       x0, y0, width, height = item.get_paths()[0].get_extents().bounds
       item.set_clip_path(plt.Rectangle((x0, y0-0.05), width, height/2,
                       transform=axviol.transData))

num_items = len(axviol.collections)
    
sns.stripplot(ax=ax[ii], y='Sample', x='Track Lifetime (hrs)', 
              data = all_data, size=4, zorder=1, edgecolor='black', linewidth=1,
              alpha=1, jitter=True, order=order_,
              palette=colors)
    
    
# Shift each strip plot strictly below the correponding volin.
for item in axviol.collections[num_items:]:
       item.set_offsets(item.get_offsets() + 0.1)
        
   
ax[ii].set_xlim(-1,13)

ax[ii].set_ylabel('')
ax[ii].tick_params(axis='x',labelsize=font_size)
ax[ii].tick_params(axis='y',labelsize=font_size)



    
#axes[0].set_ylim(0,11)
axes[0].set_xlabel('Focus lifespan (h)', fontsize=font_size)
#axes[0].set_xlabel('')
#axes[0].tick_params(axis='x',labelsize=font_size,rotation = 45)
axes[0].tick_params(axis='y',labelsize=font_size)

# for axis in ['top','bottom','left','right']:
#    axes[0].spines[axis].set_linewidth(2)

# increase tick width
axes[0].tick_params(width=2)


fig.tight_layout()
#plt.savefig(directory[:-14] + '20220816_Figure_lifespace_ceph_all.png', dpi=300)
plt.show()

################################################################
'''
q = sns.violinplot(ax=axes[1], x='Sample', y='Fluorescence_max_diff', 
              data = all_data, 
              alpha=0.25, order=order_, scale = 'area',
              cut=0, inner=None,
              palette=['royalblue','mediumseagreen','palegoldenrod','silver'])
for item in q.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0), width/2, height,
                       transform=q.transData))
num_items_q = len(q.collections)
for item in q.collections[num_items_q:]:
    item.set_offsets(item.get_offsets())

sns.stripplot(ax=axes[1], x='Sample', y='Fluorescence_max_diff', 
              data = all_data, size=5, zorder=1,
              alpha=0.4, jitter=True, 
              order=order_,
              palette=colors)

# Create narrow boxplots on top of the corresponding violin and strip plots, with thick lines, the mean values, without the outliers.
sns.boxplot(ax=axes[1], x='Sample', y='Fluorescence_max_diff', data=all_data, 
            width=0.25,order=order_, color = 'black',
            showfliers=False, showmeans=True, 
            meanprops=dict(marker='o', markerfacecolor='black',
                           markeredgecolor='black',
                           markersize=7, zorder=5),
            boxprops=dict(facecolor=(0,0,0,0), 
                          linewidth=2, zorder=4),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            medianprops=dict(linewidth=2))

axes[1].set_ylim(-1,.1)
axes[1].set_ylabel('$Intensity_f$ - $Intensity_{max}$', fontsize=font_size)
axes[1].set_xlabel('')
axes[1].tick_params(axis='x',labelsize=font_size,rotation = 45)
axes[1].tick_params(axis='y',labelsize=font_size)

'''



