# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 14:35:06 2022

@author: azaldegc
"""
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
from sklearn.mixture import GaussianMixture
from matplotlib.collections import LineCollection
import scipy.stats as stats

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)
font_size = 20
add_division_events = False
dfs = []
samples= []
division_events = [5.75, 12.83, 18.92, 25.08, 31.33, 38, 44.5, 50.82] # in frames
for file in filenames:
    
    df = pd.read_csv(file)
    dfs.append(df)
    samples.append(df['sample'].iloc[0])
    
all_data = pd.concat(dfs)
print(all_data)
samples = list(set(samples))
#samples.sort()
order_ = ['cI', 'PopTag-SL', 'PopTag-LL', 'McdB']
colors = ['darkblue', 'darkgreen', 'darkorange', 'purple']
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']

print(samples)

fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(10, 4), 
                               sharey=False, sharex=False)

if add_division_events == True:
    for xcoords in division_events:
        div_hr = (xcoords * 15) / 60
        axes[0].axvline(x=div_hr, color='black', linestyle='dashed', alpha = 0.4)
    
lines = []
for ii, sample in enumerate(order_):
    
    x = all_data[all_data['sample']==sample]['time'].to_numpy()
    y = all_data[all_data['sample']==sample]['trace average'].to_numpy()
    error = all_data[all_data['sample']==sample]['trace stdev'].to_numpy()
    points = np.array([x,y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    n_seq = all_data[all_data['sample']==sample]['N'].to_numpy()
    norm_n = [n/max(n_seq) for n in n_seq]
    norm = plt.Normalize(0, max(norm_n))
    lc = LineCollection(segments, norm=norm,
                      
                        cmap=cmaps[ii], label=order_[ii])
    lc.set_array(norm_n)
    lc.set_linewidth(4)
    line = axes[0].add_collection(lc)
    lines.append(lc)
  #  axes[0].plot(x, y, color = colors[ii],linewidth=3, label=sample)
    axes[0].fill_between(x,y-error, y+error, alpha=0.20, 
                         linewidth=1,color=colors[ii])
    
leg = axes[0].legend(bbox_to_anchor=(1.4,1), ncol=1)
leg.legendHandles[0].set_color('darkblue')
leg.legendHandles[1].set_color('darkgreen')

leg.legendHandles[2].set_color('darkorange')
leg.legendHandles[3].set_color('purple')
   
axes[0].set_xlim(0,10)
axes[0].set_ylim(0,1.1)
axes[0].set_xlabel('Time (h)', fontsize=font_size)
axes[0].set_ylabel('Total focus intensity',fontsize=font_size)
axes[0].tick_params(axis='x',labelsize=font_size)
axes[0].tick_params(axis='y',labelsize=font_size)
fig.tight_layout()
plt.savefig(directory[:-11] + '20220815_Figure_cephtreat_averagetracesall.png', dpi=300)
plt.show()

    