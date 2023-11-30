# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:49:49 2023

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
font_size = 9
dfs = []

order_ = ['PopTag_SL', 'PopTag_LL', 'McdB']
#colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]
for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
 
df = pd.concat(dfs)
df = df.reset_index()




print(df)
# perform t_test
tf = df[df['time'] == 1]
group1 = tf[tf['sample'] == 'PopTag_LL']
group2 = tf[tf['sample'] == 'PopTag_SL']
   
print(ttest_ind(group1['avg_cell_fluor'], group2['avg_cell_fluor'], equal_var=False))

for sample in order_:
    print(sample)
    dff = df[df['sample']==sample]
    
    pre = dff[dff['time']==0]['avg_cell_fluor']
    
    print("n, pre Mean + stdev", len(pre), np.mean(pre), np.std(pre))
    
    post = dff[dff['time']==1]['avg_cell_fluor']
    print("n, post Mean + stdev",len(post), np.mean(post), np.std(post))






fig, axes = plt.subplots(figsize=(3, 3.1), 
                         dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'


    
axia = sns.boxplot(x="sample", y="avg_cell_fluor", data=df, hue='time',order=order_,
                     palette=['darkred', "gray"], linewidth=1, whis=[5, 95], 
                     showfliers = False)
ax = sns.stripplot(x="sample", y="avg_cell_fluor", data=df, hue='time',order=order_,
                       palette=['darkred', "gray"],dodge=True,jitter=True,
                        edgecolor ='black',linewidth=1,alpha=1,
                       size=5)
handles, labels = axia.get_legend_handles_labels()
axia.legend(handles[:2], ['Pre', 'Post'], title='', loc='upper left', 
            ncol=2, frameon=False, fontsize=font_size)
axes.set_ylabel('Fluorescence Conc. (a.u.)', 
                          fontsize=font_size)
axes.set_yticks(axes.get_yticks()[::2])
axes.set_ylim(0, 3500)
axes.set_xticks(ticks=axes.get_xticks(), labels=['$\mathregular{PopTag^{SL}}$', '$\mathregular{PopTag^{LL}}$', '$\mathregular{McdB^{}}$'])
axes.set_xlabel('', fontsize=font_size)
axes.tick_params(axis='x',labelsize=font_size)
#xticks = ['', 'No Focus']
#axes.set_xticklabels(xticks)
axes.tick_params(axis='y',labelsize=font_size)
#axes.get_legend().remove()
axes.set_title("Condensate Dissolution", fontsize=font_size)

fig.tight_layout()
plt.savefig(directory[:-5] + 'fluorescence_at_dissolution.svg', dpi=300)
plt.show()