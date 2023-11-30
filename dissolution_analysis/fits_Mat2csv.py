# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:37:30 2021

@author: azaldegc


Script to extract the tracks and fits array from the SMALL-LABS output fits.mat
file. Converts tracks into dataframe and saves as csv. Converts fits into dataframe
and saves as csv.
"""

import h5py
import sys
import numpy as np
import pandas as pd
import glob


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# foldername that contains all of the fits.mat files
directory = sys.argv[1]
# pull in all of the fits.mat files
mat_files = filepull(directory)
do_tracks = True

# for each fits.mat file
for file in mat_files:
    
    # init fits storage lists
    fits_store = []
    fits_names = []

    # read the .mat file
    with h5py.File(file, 'r') as f:
            
            # get fits structures key names
            fits = f['fits'].keys()
            
            # store the data from each key
            for group in fits:
                # store key name ie array column title
                fits_names.append(group)
                
                # store data
                fits_store.append(f['fits'][group][:])
                
            # assign tracks array to var
            tracks = f['tracks'][:]
         
    # have to restructure fits data since its in 1D array
    fits_data = [[] for ii in range(len(fits_store))]
    # for each key (or column)
    for ii,feat in enumerate(fits_store):
        # for each array in column. Should only be one
        for arr in feat:
            # for each element in the one array
            for el in arr:
                # store value in appropriate list
                fits_data[ii].append(el)
    # transpose fits data 
    fits_data_transpose = np.array(fits_data).T
    # convert fits data into dataframe and save
    fits_df = pd.DataFrame(fits_data_transpose, columns=fits_names)
    fits_df.to_csv(file[:-4]+'_fits.csv', index=False)
    
    # transpose, convert, and save tracks data
    if do_tracks == True:
        if tracks.shape == (2,):
            tracks = np.zeros((6,1))
        tracks_name = ['FRAME_N','LOC_R','LOC_C','TRACK_N','ROI_NUM','MOL_ID']
        tracks_df = pd.DataFrame(tracks.T, columns=tracks_name)
        tracks_df.to_csv(file[:-4]+'_tracks.csv', index=False)
        
    print(file)