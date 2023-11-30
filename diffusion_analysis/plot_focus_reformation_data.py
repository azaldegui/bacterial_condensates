# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 16:11:03 2023

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

dfs = []

#samples = ['mCherry', 'mCherry-D8N', 'mNeonGreen']
#colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]

for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    df = df[df['Sample']!='Mother']
 

    df = df.reset_index()
    
    print(df)
    # perform t_test
    tf = df
    group1 = tf[tf['Sample'] == 'Inherited']
    group2 = tf[tf['Sample'] == 'Absent']
       
    print(ttest_ind(group1['Normalized Cellular Intensity'], group2['Normalized Cellular Intensity'], equal_var=False))
    
    
    daughter_1 = df[df['Sample']=='Inherited']
    print("daughter with focus: ", np.mean(daughter_1['Normalized Cellular Intensity']), np.std(daughter_1['Normalized Cellular Intensity']))
    daughter_2 = df[df['Sample']=='Absent']
    print("daughter without focus: ", np.mean(daughter_2['Normalized Cellular Intensity']), np.std(daughter_2['Normalized Cellular Intensity']))





    fig, axes = plt.subplots(figsize=(3, 3), 
                         dpi=300)
    
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'


    font_size = 9
    
    sns.set(style="whitegrid")


    #ax = sns.boxplot(ax = axes[0], x="Sample", y="Normalized Average Intensity", data=df, showfliers = False)
    #ax = sns.swarmplot(ax = axes[0], x="Sample", y="Normalized Average Intensity", data=df, color=".25")
    
    ax = sns.boxplot(x="Sample", y="Normalized Cellular Intensity", data=df, 
                     palette=['darkred', "gray"], linewidth=1, 
                     showfliers = False)
    ax = sns.stripplot(x="Sample", y="Normalized Cellular Intensity", data=df, 
                       hue='Sample',palette=['darkred', "gray"],
                       dodge=False,jitter=True, edgecolor ='gray',linewidth=1,
                       size=5)
    axes.set_ylabel('Normalized Fluorescence Intensity', 
                          fontsize=font_size)
    axes.set_yticks(axes.get_yticks()[::2])
    axes.set_xlabel('', fontsize=font_size)
    axes.tick_params(axis='x',labelsize=font_size)
  #  axes.set_ylim(.4, .6)
    xticks = ['Focus', 'No Focus']
    axes.set_xticklabels(xticks)
    axes.tick_params(axis='y',labelsize=font_size)
    axes.get_legend().remove()
    axes.set_title("PopTag_SL", fontsize=font_size)
    

    fig.tight_layout()   
    plt.savefig(directory[:-12] + 'divreform_plot.svg', dpi=300) 
    plt.show()