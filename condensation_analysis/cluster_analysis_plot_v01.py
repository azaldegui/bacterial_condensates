# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:42:22 2022

@author: azaldegc
"""



import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys


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

for file in filenames:
    
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
    samples.append(df['Label'].iloc[0])
    
all_data = pd.concat(dfs)
samples = list(set(samples))
samples.sort()
order_ = ['cI', 'PopTag-SL', 'PopTag-LL', 'McdB', 'McdB-delPS', 'mCherry']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'magenta']
print(samples)

fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(8,4), 
                               sharey=True, sharex=False, dpi = 300)
ax = axes.ravel()

threshs =['cluster 0.5']
thresh = [0.5]

for ii in range(1):
    
    
    # determine the mean and std for cI and mCherry datasets
    # cI_data = all_data[all_data['Label']=='cI'][threshs[ii]].to_numpy()
    # cI_mean, cI_ste = np.mean(cI_data), np.std(cI_data)#/np.sqrt(len(cI_data))
    mCherry_data = all_data[all_data['Label']=='mCherry'][threshs[ii]].to_numpy()
    mCherry_mean, mCherry_ste = np.mean(mCherry_data)*100, np.std(mCherry_data)*100#/np.sqrt(len(mCherry_data))
    cutoff = mCherry_mean + mCherry_ste
    
    # plot the rain cloud plot
    all_data[threshs[ii]] = all_data[threshs[ii]].apply(lambda x: x*100)
    axviol = sns.violinplot(ax=ax[ii], x=threshs[ii], y='Label', data=all_data,
                    color='white', order=order_, zorder=0,
                    linewidth=2, edgecolor='black', scale='width', scale_hue=True,
                    cut=0, inner=None)
    
    for item in axviol.collections:
        x0, y0, width, height = item.get_paths()[0].get_extents().bounds
        item.set_clip_path(plt.Rectangle((x0, y0-0.05), width, height/2,
                       transform=axviol.transData))

    num_items = len(axviol.collections)
    
    sns.stripplot(ax=ax[ii], y='Label', x=threshs[ii], 
              data = all_data, size=4, zorder=1, edgecolor='black', linewidth=1,
              alpha=1, jitter=True, order=order_,
              palette=colors)
    
    
    # Shift each strip plot strictly below the correponding volin.
    for item in axviol.collections[num_items:]:
        item.set_offsets(item.get_offsets() + 0.1)
        
    # add the vertical lines for the mCherry mean and stdev
    ax[ii].axvline(x=mCherry_mean-mCherry_ste, color='black', linestyle='--', zorder=0)
    ax[ii].axvline(x=mCherry_mean, color='black', linestyle='-', zorder=0)
    ax[ii].axvline(x=mCherry_mean+mCherry_ste, color='black', linestyle='--', zorder=0)
    
    
    ax[ii].set_xlim(0,110)
    ax[0].set_xlabel('Condensation percentage (%)', 
                          fontsize=font_size)
    ax[ii].set_ylabel('')
    ax[ii].tick_params(axis='x',labelsize=font_size)
    ax[ii].tick_params(axis='y',labelsize=font_size)
    
    percent_left_all = []
    for samp in order_:
        
        data = all_data.loc[all_data['Label'] == samp]
        condensation_data = data[threshs[ii]]
        print(len(condensation_data))
        tab = 0
        for val in condensation_data:
            if val >= cutoff:
                tab += 1
        percent_left = (tab / len(condensation_data))
        percent_left_all.append(percent_left)
    
    ax[1].barh(np.arange(len(percent_left_all)), percent_left_all, 
               align='center', color= colors, linewidth=3, edgecolor='black',
               alpha = 0.95)   
    ax[1].set_xlabel('Fraction above cutoff', fontsize=font_size)   
    ax[1].set_xlim(0,1) 
    ax[1].tick_params(axis='x',labelsize=font_size)
    ax[1].tick_params(axis='y',labelsize=font_size)
   # ax[1].set_yticks(np.arange(len(percent_left_all)), labels)
  


fig.tight_layout()
plt.savefig(directory[:-5] + '20220816_Figure_condensation_all.png', dpi=300)
plt.show()




n_bins = 25
fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(8, 4), 
                               sharey=False, sharex=False)
axel = axes.ravel()
for ii, thresh in enumerate(threshs[1:3]):
    
    for kk, sample in enumerate(order_):
        cluster_coeffs = all_data[all_data['Label']==sample][thresh].to_numpy()
        axel[ii].hist(cluster_coeffs, density = True, bins = n_bins,
                       histtype = 'step', color = colors[kk],
                       linewidth=3)
    
    axel[ii].set_title(thresh, fontsize=font_size)  
    axel[ii].set_xlabel('Condensation parameter', fontsize=font_size)    
    axel[ii].set_ylabel("Density", fontsize=font_size)   
    axel[ii].tick_params(axis='x',labelsize=font_size)
    axel[ii].tick_params(axis='y',labelsize=font_size)
#fig.tight_layout()
#plt.show()
