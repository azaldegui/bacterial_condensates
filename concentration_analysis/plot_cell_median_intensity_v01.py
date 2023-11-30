# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:08:17 2023

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


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory = sys.argv[1]
filenames = filepull(directory)

dfs = []
samples=  ['cI_agg', 'PopTag_SL', 'PopTag_LL', 'McdB', 'McdB_sol', 'mCherry-']
#samples = ['mCherry', 'mCherry-D8N', 'mNeonGreen']
colors = ['royalblue', 'darkgreen', 'darkorange', 'purple', 'gold', 'gray']
#colors = ['black', 'darkmagenta', 'seagreen' ]
for file in filenames:
    
    df = pd.read_csv(file,index_col=None)
    dfs.append(df)
 
all_data = pd.concat(dfs)
all_data = all_data.reset_index()

# make a plot for each sample
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(4.75, 2.75), sharex=True, sharey=True, dpi=100)
ax = axes.ravel()

plt.rcParams.update({'font.size': 8})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'


font_size = 8
for jj, sample in enumerate(samples):
    
    
    
    
    # assign the sample dataset
    sample_data = all_data[all_data['Label']==sample]
    
    time = [0,1, 2, 3, 4, 5]
    # make three plots (total cell intensity, cytoplasmid intensity, focus intensity)
    #fig, axes = plt.subplots(nrows=len(time), figsize=(4, 8), sharex=True, dpi=100)
    #ax = axes.ravel()
    
    
    
    # to make a df to plot using sns
    data_store = []
    
    
    print(sample)
    for kk, t in enumerate(time[:]):
        data_for_plot = []
        
        
        
        time_data = sample_data[sample_data['Time'] == t]
        
        
        # determine % of timepoint with focus
        plot_threshold = .15
        fraction_focus = len(time_data[time_data['Focus'] > 0]) / len(time_data)
        print("fraction with focus", fraction_focus)
        
        mean_cell_intensity = []
        mean_cytoplasm_intensity = []
        mean_foci_intensity = []
        
        no_foci_cell_intensity = []

        # for each cell
        for index, row in time_data.iloc[:].iterrows():
        
            cell_area = json.loads(row['Cell Area'])[0]
    
            cell_intensity = json.loads(row['Cell Intensity'])[0]
            if row['Focus'] > 0: 
                
        
                mean_cell_intensity.append(cell_intensity / cell_area)
                
                if fraction_focus >= plot_threshold:
                    data_store.append((t, 'Yes', cell_intensity / cell_area ))
                
                cyto_area = json.loads(row['Cytoplasm Area'])[0]
                cyto_intensity = json.loads(row["Cytoplasm Intensity"])[0]
            
                mean_cytoplasm_intensity.append(cyto_intensity / cyto_area)
            
            
                foci_area = json.loads(row["Focus Area"])
                foci_intensity = json.loads(row['Focus Intensity'])
                
                for ff, focus in enumerate(foci_intensity):
                    
                    mean_foci_intensity.append(focus / foci_area[ff])
            
            elif row['Focus'] == 0:
                
                no_foci_cell_intensity.append(cell_intensity / cell_area)
                
                if 1 - fraction_focus >= plot_threshold:
                    data_store.append((t, 'No', cell_intensity / cell_area ))
        print('time', t)      
        print("Focus median: ", np.median(mean_cell_intensity))
        print("No Focus median: ", np.median(no_foci_cell_intensity))
        print()

        '''
        # plot histograms
        n_bins = 25
        ax[kk].hist(no_foci_cell_intensity, bins='auto', alpha=0.9, label='No focus', 
                    histtype='step', color = 'black', linewidth=2)
        #ax[kk].hist(mean_foci_intensity, bins=n_bins, alpha=0.5, label='Focus', histtype='step')
        ax[kk].hist(mean_cell_intensity, bins='auto', alpha=0.9, label='Focus', 
                    histtype='step', color = 'red', linewidth=2)
        #ax[kk].set_xscale('log')
        #ax[kk].hist(mean_cytoplasm_intensity, bins=n_bins, alpha=0.5, label='Cyto', histtype='step')
        
        
        ax[kk].set_ylabel("Counts")
        ax[kk].set_xlim(0, 10000)
        ax[kk].legend()
        print()
        '''
    #ax[0].set_title(sample)   
    #ax[len(time)-1].set_xlabel('Fluorescence conc. [a.u.]')   
        
        
            
    #fig.tight_layout()
    #plt.show()
    
    
    plt.rcParams.update({'font.size': 15})
    
    data_df = pd.DataFrame(data_store)
    data_df.columns = ['Time (h)', 'Focus', 'Fluorescence conc.']
    print(data_df)
    
    
    
    axia = sns.lineplot(ax=ax[jj], data = data_df, y="Fluorescence conc.", x="Time (h)", hue="Focus",  
                      errorbar=('ci',95), linewidth=1, marker='o',markersize=5,markeredgecolor='black',
                      palette=['gray', "darkred",],
                      )
   
    handles, labels = axia.get_legend_handles_labels()
    axia.legend(handles[:2], labels[:2], loc='upper left', ncol=1, frameon=False)
    ax[jj].set_ylim(0, 4000)
    #ax[jj].set_ylim(10**2, 6000)
    #ax[jj].set_yscale('log')
    ax[jj].set_ylabel('Fluorescence conc.', 
                          fontsize=font_size)
    ax[jj].set_xlabel('Time post-induction (h)', fontsize=font_size)
    ax[jj].tick_params(axis='x',labelsize=font_size)
    ax[jj].tick_params(axis='y',labelsize=font_size)
    
fig.tight_layout()   
plt.savefig(directory[:-5] + 'fluorescenceconc_v_time.svg', dpi=300) 
plt.show()