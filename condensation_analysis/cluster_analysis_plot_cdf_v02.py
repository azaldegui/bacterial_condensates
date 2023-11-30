# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:05:30 2022

@author: azaldegc

Plot CDF of condensation parameter for proteins at different threshold values
"""


import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import scipy.stats as ss
from scipy.special import kolmogorov


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)
font_size = 12
bootstrap = 100
boots_frac = .7
sample_n = 250
dfs = []
samples= []

for file in filenames:
    
    df = pd.read_csv(file)
    dfs.append(df)
    samples.append(df['Label'].iloc[0])
    
all_data = pd.concat(dfs)
samples = list(set(samples))
samples.sort()
order_ = ['cI', 'PopTag-SL', 'PopTag-LL', 'McdB', 'McdB-delPS', 'mCherry']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 
          'darkgoldenrod', 'magenta']
print(samples)

fig, axes = plt.subplots(ncols=5,nrows=2, figsize=(15, 5), 
                               sharey=False, sharex=False)
ax = axes.ravel()

threshs =['cluster 0.1','cluster 0.3', 'cluster 0.5', 'cluster 0.7','cluster 0.9']
thresh = [0.1,0.3, 0.5, 0.7,0.9]
cutoff = []
mcherry_data = all_data[(all_data['Label']=='mCherry')]
for tt in range(5):
    mcherry_stdev = np.std(mcherry_data[threshs[tt]].to_numpy())
    print(mcherry_stdev)
    cutoff.append(float(thresh[tt]) + mcherry_stdev )
print(cutoff)

#cutoff = [0.1, 0.3, 0.5, 0.7, 0.9]

sets = [[1,2,3],[4,5,6],[7,8,9,10]]
for ii in range(5):
    
    means = []
    error = []
    
    for jj,label in enumerate(order_[0:]):
        
        
        sfs_arr = []
        x_arr = []
        for set_ in sets:
            
            data = all_data[(all_data['Label']==label) & 
                            all_data['FILE_ID'].isin(set_)]
           
            cp = data[threshs[ii]].to_numpy()
            
            x = np.sort(np.random.choice(cp, replace=True, size=sample_n))
           
            target = ss.norm(0,1)
            cdf = target.cdf(x)
            ecdf = np.arange(len(x), dtype=float) / len(x)
           
            sf = 1 - ecdf
           
            sfs_arr.append(sf)
            x_arr.append(x)
           
           
          #  ax[ii].plot(x, sf, color=colors[jj], label=label, 
           #             linewidth=1)
      
  
        sfs_array = np.asarray(sfs_arr)
        x_array = np.asarray(x_arr)
        
        sfs_mean = np.average(sfs_array.reshape(-1, sample_n), axis=0)
        x_mean = np.average(x_array.reshape(-1, sample_n), axis=0)
       # print(sfs_mean)
     
        if len(np.where(x_mean >= cutoff[ii])[0]) >= 1:
            co_val = np.where(x_mean >= cutoff[ii])[0][0]
            x_SEM = x_array.std(axis=0) / 3**0.5 
            x_std = x_array.std(axis=0)
         
       #     print(x_mean[co_val], x_SEM[co_val])
            means.append(sfs_mean[co_val])
            error.append(x_std[co_val])
        else:
            means.append(0)
            error.append(0)   
            x_SEM = x_array.std(axis=0) / 3**0.5 
            x_std = x_array.std(axis=0)
        print()
       
        ax[ii].set_ylim(0,1.)  
        ax[ii].plot(x_mean, sfs_mean, color=colors[jj], label=label, 
                   linewidth=2)
        ax[ii].fill_between(x_mean, sfs_mean - x_SEM,
                       sfs_mean + x_SEM, alpha=0.5, 
                      color=colors[jj])  
        ax[ii].axvline(x=cutoff[ii], color='k', linestyle='dashed', 
                       linewidth=1, alpha=0.75)
    error = [x*1 for x in error]
    ax[ii+5].errorbar(order_, means, error, linestyle='None', marker='o',
                      ms=3, c='k', capsize=3, elinewidth=1)
    ax[ii+5].set_xticklabels(order_, rotation=90)
        
        
    #ax[ii].set_xlim(0,1.) 
    ax[ii].tick_params(axis='y',labelsize=font_size)
    ax[ii].tick_params(axis='x',labelsize=font_size)
    ax[ii+5].tick_params(axis='y',labelsize=font_size)
    ax[ii+5].tick_params(axis='x',labelsize=font_size)
    ax[ii].set_xlabel('Condensation param. (t={})'.format(thresh[ii]), 
                      fontsize=font_size)
        
ax[4].legend(loc='upper left', prop={'size': 8},  bbox_to_anchor=(1, 1))
    
ax[0].set_ylabel('Fraction remaining', fontsize=font_size)    
ax[5].set_ylabel('Fraction remaining', fontsize=font_size)  
fig.tight_layout()
plt.show()
