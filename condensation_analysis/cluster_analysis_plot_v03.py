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
import scipy.stats as ss


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def norm2mCherry(C, mCherry):
    
    return (C - mCherry) / (1 - mCherry)

directory = sys.argv[1]
filenames = filepull(directory)
font_size = 15
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
order_ = ['cI_agg', 'PopTag_SL', 'PopTag_LL', 'McdB', 'McdB_PS-']#, 'mCherry']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold']#, 'magenta']
print(samples)

fig, axes = plt.subplots(figsize=(5,4), dpi = 300)


threshs =['cluster 0.7']
thresh = [0.7]


# determine the stats for mCherry (no focus) dataset
mCherry_data = all_data[all_data['Label']=='mCherry']#[threshs[ii]]
mCherry_nofocus_data = all_data[all_data['Focus']==0][threshs[0]].to_numpy()
mCherry_mean, mCherry_std = np.mean(mCherry_nofocus_data), np.std(mCherry_nofocus_data)
cutoff = mCherry_mean - mCherry_std #lower bound to normalize to
print(mCherry_mean, mCherry_std)

# using the standard error determine the lower bound for measurable condensation
cutoff_adj = norm2mCherry(mCherry_mean + mCherry_std, cutoff)*100#(np.std(mCherry_nofocus_data)/np.sqrt(len(mCherry_nofocus_data))), cutoff)*100


all_data_nofocus = all_data.copy()
all_data_focus = all_data.copy()
    
# plot the rain cloud plot
all_data_focus[threshs[0]] = all_data_focus[all_data_nofocus['Focus']>0][threshs[0]].apply(lambda x: 100*norm2mCherry(x,cutoff))

# normalize to mcherry mean - mcherry std (that is the new 0)
# exclude adjusted values below 0

all_ccs = []
for sample in order_:
    
    ccs_ = all_data_focus[all_data_nofocus['Focus']>0]
   
    ccs = ccs_[ccs_['Label']==sample][threshs[0]].to_numpy()
    print(sample,'n:', len(ccs) )
    #print(ccs)
    print("condensation coeff: ", np.median(ccs), np.std(ccs), np.std(ccs) / (len(ccs))**0.5)

    for ii in range(len(all_ccs)):
    
        print(ss.ttest_ind(ccs, all_ccs[ii], equal_var=False))
    all_ccs.append(ccs)



all_data_focus = all_data_focus[all_data_focus[threshs[0]] > 0]


axviol = sns.violinplot(ax=axes, x=threshs[0], y='Label', data=all_data_focus,
                    color='white', order=order_, zorder=0,
                    linewidth=2, edgecolor='black', scale='width', scale_hue=True,
                    cut=0, inner=None)
    
for item in axviol.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0-0.05), width, height/2,
                       transform=axviol.transData))

num_items = len(axviol.collections)
    
sns.stripplot(ax=axes, y='Label', x=threshs[0], 
              data = all_data_focus, size=4, zorder=1, edgecolor='black', linewidth=1,
              alpha=0.15, jitter=True, order=order_,
              palette=colors)
    
    
# Shift each strip plot strictly below the correponding volin.
for item in axviol.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.1)
    


'''
####    
# plot the rain cloud plot
all_data_nofocus[threshs[0]] = all_data_nofocus[all_data_nofocus['Focus']==0][threshs[0]].apply(lambda x: 100*norm2mCherry(x,cutoff))

all_data_nofocus = all_data_nofocus[all_data_nofocus[threshs[0]] > 0]

axviol = sns.violinplot(ax=ax[1], x=threshs[0], y='Label', data=all_data_focus,
                    color='white', order=order_, zorder=0,
                    linewidth=2, edgecolor='black', scale='width', scale_hue=True,
                    cut=0, inner=None)
    
for item in axviol.collections:
    x0, y0, width, height = item.get_paths()[0].get_extents().bounds
    item.set_clip_path(plt.Rectangle((x0, y0-0.05), width, height/2,
                       transform=axviol.transData))

num_items = len(axviol.collections)
    
sns.stripplot(ax=ax[1], y='Label', x=threshs[0], 
              data = all_data_nofocus, size=4, zorder=1, edgecolor='black', linewidth=1,
              alpha=1, jitter=True, order=order_,
              palette=colors)
    
    
# Shift each strip plot strictly below the correponding volin.
for item in axviol.collections[num_items:]:
    item.set_offsets(item.get_offsets() + 0.1)   
    
'''

for ii in range(1):
    # add the vertical lines for the mCherry mean and stdev
    axes.axvspan(0,cutoff_adj, color='black', alpha=0.25, zorder=1)

    
    
    axes.set_xlim(0, 120)
  
    
    axes.set_xlabel('Condensation coefficient', 
                          fontsize=font_size)
    axes.set_ylabel('')
    axes.tick_params(axis='x',labelsize=font_size)
    axes.tick_params(axis='y',labelsize=font_size)
    
    


fig.tight_layout()
#plt.savefig(directory[:-5] + '20221024_Figure_condensation_all_30.png', dpi=300)
plt.show()


