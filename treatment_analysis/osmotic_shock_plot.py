# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:53:17 2023

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


protein = 'McdB' #, 'PopTag_SL', 'PopTag_LL', 'McdB']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple']

ctrl = ['ctrl_0min', 'ctrl_15min', 'ctrl_30min']
treat = ['salt', 'wash_15min', 'wash_30min']


order = [ctrl, treat]

colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]
for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
 
df = pd.concat(dfs)
df = df.reset_index()

out_data = []
for ii, samp_set in enumerate(order):
    
    for jj, sample in enumerate(samp_set):
    
        sample_data = df[df['Label']==sample]
        print(sample)
        t = jj*15
   
        n_files = sample_data['Replicate'].max()
        print(n_files)
                                                                         
        for rep in range(n_files):
            rep_data = sample_data[sample_data['Replicate']==rep+1]
            print(rep)
        
            n_cells_focus = len(rep_data[rep_data['Focus']>0].to_numpy())
            print(sample, n_cells_focus, len(rep_data),(n_cells_focus / len(rep_data))*100)
               

           
            out_data.append([ii, t, sample, (n_cells_focus / len(rep_data))*100]) 
        print()
        


newdf = pd.DataFrame(out_data)
newdf.columns = ["type", 'time', 'Sample', 'Percent cells with focus']

print(newdf)
newdf.to_csv(directory[:-5] + protein + '_' +
         '_osmoticdata.csv', index = False)   

fig, axes = plt.subplots(figsize=(2.5, 2), 
                         dpi=200)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'



axia = sns.lineplot(x="time", y="Percent cells with focus", data=newdf, hue='type',
                     palette=['gray', 'darkblue'], linewidth=1, errorbar='sd',
                     err_style='bars',
                     marker='o',markersize=3,markeredgecolor='black',)
'''
ax = sns.stripplot(x="Sample", y="Percent cells with focus", data=newdf, order=order_,
                       palette=['purple'],
                        edgecolor ='black',linewidth=1,alpha=1,
                       size=5)
'''
handles, labels = axia.get_legend_handles_labels()
axia.legend(handles[:2], ['Control', 'Treated'], title='', loc='upper left', 
            ncol=2, frameon=False, fontsize=font_size)
axes.set_ylabel('Cells with a fluorescent focus (%)', 
                          fontsize=font_size)
#axes.set_yticks(axes.get_yticks()[::2])
axes.set_ylim(0, 100)
#axes.set_xticks(ticks=axes.get_xticks(), labels=['Ctrl', '+ KCl', 'wash KCl'])
axes.set_xlabel('Time (min)', fontsize=font_size)
axes.tick_params(axis='x',labelsize=font_size)
#xticks = ['', 'No Focus']
#axes.set_xticklabels(xticks)
axes.tick_params(axis='y',labelsize=font_size)
#axes.get_legend().remove()
#axes.set_title("Condensate Dissolution", fontsize=font_size)

fig.tight_layout()
plt.savefig(directory[:-5] + protein + '_osmotic_plot.svg', dpi=300)
plt.show()
