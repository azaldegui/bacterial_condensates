# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:25:55 2022

@author: azaldegc
"""

import sys
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# user defined parameters
font_size = 15

# load files
directory = sys.argv[1]
filenames = filepull(directory)
dfs = []
filenames.sort()

data = []

# for each file
for file in filenames:
    
    df = pd.read_csv(file) # read file and convert to dataframe
    
    df.transpose() # transpose dataframe
  
    # define background signal (Bsig)
    bg_arr = df.iloc[-2, 1:].to_numpy()
    
    # define reference signal to account for photobleaching (PBsig)
    ref_arr = df.iloc[-1, 1:].to_numpy()
    
    # for each frap curve perform corrections
    for ii in range(len(df) - 2):
        
        # find protein label and index
        label = df.iloc[ii,0].partition('_')
        name = label[0]
        index =  label[2]
        out = [name, index]
        
        # frap signal 
        foci_arr = df.iloc[ii, 1:].to_numpy()
        
        # background correct frap signal
        bl_corrected = foci_arr - bg_arr
        
        # background correct ref signal
        ref_corrected = ref_arr - bg_arr
        
        # normalize corrected foci signal to corrected ref signal
        frap_arr = np.divide(bl_corrected, ref_corrected)
        
        
        # below can determine whether to normalize only to the pre-bleach intensity
        # or also normalize to the post-bleach intensity
        
        # normalize corrected foci signal to pre-bleach and post bleach intensity
        normed_frap = (frap_arr - frap_arr[1]) / (frap_arr[0] - frap_arr[1])
       
        # only normalize pre-bleach intensity
      #  normed_frap = frap_arr / frap_arr[0]
        
        # save corrected curve into list
        out.extend(normed_frap)
        data.append(out)

# convert results into dataframe and save
out_df = pd.DataFrame(data)
out_df.columns = ['Sample', 'Index'] + list(df.columns)[1:]
print(out_df)        
out_df.to_csv(directory[:-5] + 'FRAP_data_corrected_normalized.csv', index = False)