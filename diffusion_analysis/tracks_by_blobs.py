# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 01:41:10 2022

@author: azaldegc
"""



# import modules
import sys
import numpy as np
import glob
from PIL import Image
import tifffile as tif
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

from matplotlib import cm
import matplotlib.patches as patches
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from matplotlib_scalebar.scalebar import ScaleBar
from math import log
from scipy.optimize import curve_fit



# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# read csv
def csv2df(file):
    '''
    Reads a .csv file as a pandas dataframe
    
    input
    -----
    file: string, csv filename
    
    output
    -----
    df: pd.dataframe
    '''
    df = pd.read_csv(file, header = 0, index_col=False)
    return df


def total_tracks(df):
    n_tracks = df['TRACK_N'].max()
   
    return int(n_tracks)

# Separate tracks by ID into dictionary
def tracks_df2dict(df, n_tracks, min_length):
    
    tracks_dict = {} # initialize empty dictionary
    
    for ii in range(n_tracks): # for each track
        
        track = df.loc[df.TRACK_N == (ii)] # assign current track to var
        track_length = len(track) # find track length (n localizations)
   
        # if track length > or = min_length then store in dict
        if track_length >= min_length: 
            tracks_dict["track_{0}".format(ii)] = df.loc[df.TRACK_N == (ii)]
                  
    tracks_kept = len(tracks_dict) # how many track have min_length?
    #print("Total tracks to analyze: {}".format(tracks_kept)) # state it

    return tracks_dict

def create_circle_mask(h,w, center, radius):
    
    Y, X =np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def logblob(img, plot_fig = False):
    
    img = filters.gaussian(img,2)
    
    #mask = 
    masked_img = np.zeros((img.shape[0],img.shape[1]))
    
    img_norm = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    
    
    blobs = feature.blob_log(img_norm, min_sigma=2, max_sigma=9, threshold=0.05,
                             overlap=0.80)
    
    
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(7, 3), dpi = 200,
                             sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img_norm, cmap='gray')
    ax[1].imshow(img_norm, cmap='hsv')
    for ii, blob in enumerate(blobs):
        ii += 1
        y, x, r = blob
        c = plt.Rectangle((x-r,y-r), height=r*2, width=r*2, color='black', linewidth=1, fill=False)
      #    print(c.get_corners())
        ax[1].add_patch(c)
        
        mask = create_circle_mask(img.shape[0], img.shape[1], center = (x,y), radius=r)
        masked_img[mask] = ii
   
    masked_img = segmentation.clear_border(masked_img)
        
        
    ax[2].imshow(masked_img, cmap='gray')    
        
    fig.tight_layout()
    if plot_fig == True:
        plt.show()
    
    return masked_img, img_norm


def check_blobs(img_norm, mask, tracks, n_tracks, min_tracks = 5, plot_fig = False):
    
    labels, n_labels = measure.label(mask, background=0, return_num=True)
    regions = measure.regionprops(labels)
    
    new_mask = mask.copy()
    
    for reg in regions:
        
        track_in_blob_count = 0
        
     #   print(reg.label)
        for tr in range(n_tracks):
            
            track = tracks.loc[tracks.TRACK_N == (tr+1)]
            track_len = len(track)
            
            
            # keep count of locs within blob
            loc_counter = 0
            # for each loc
            for mm, loc in enumerate(range(len(track))):
                
                # coordinate of loc
                x = int(track.iloc[mm, 2])
                y = int(track.iloc[mm, 1])
                
                # if coordinate is in blob increase counter
                if mask[y][x] == reg.label:
                    loc_counter += 1
          #  print("percent of track", (loc_counter/track_len))
            if (loc_counter/track_len) >= 0.80 and track_len >=7:
                track_in_blob_count += 1
            #print(track_in_blob_count)
     #   print("tracks in blob", track_in_blob_count)   
        if track_in_blob_count < min_tracks: 
            
            new_mask[new_mask == reg.label ] = 0
            
    if plot_fig == True: 
        fig, axes = plt.subplots(ncols=3,nrows=1, figsize=(7, 3), dpi = 200,
                                 sharex=True, sharey=True)
    
        ax = axes.ravel()
        ax[0].imshow(img_norm, cmap='gray')
        ax[0].contour(new_mask, zorder=1, levels=1,
                      linestyles='dashed', linewidth=1, colors='cyan')
        ax[1].imshow(mask, cmap='gray')
        #plt.colorbar(ccb, cax=ax[0])
        
        ax[2].imshow(new_mask, cmap='gray')
        
        #ax[0].set_axis_off()
        #ax[1].set_axis_off()
        #ax[2].set_axis_off()
    
    
        fig.tight_layout()
        plt.show()
    
    return new_mask
        
    
    


def findblobs(img, norm_thresh = 0.50, area_thresh_min = 16, area_thresh_max = 1200,
              eccent_thresh = 0.98, plot=False):
    
    img = filters.gaussian(img, 1.)
    img_norm = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    
    

    lower_mask = img_norm > norm_thresh
    upper_mask = img_norm <= 1.0
    mask = upper_mask * lower_mask
  
    
    
    
    local_thresh = filters.threshold_local(img_norm, 5)
    mask = img_norm > local_thresh
    
    
    
    cell_blobs, n_cell_blobs = measure.label(mask, 
                                             background = 0,return_num = True)
   
    properties =['area','bbox','convex_area','bbox_area',
                 'major_axis_length', 'minor_axis_length',
                 'eccentricity']
    
    cell_blob_props = pd.DataFrame(measure.regionprops_table(cell_blobs, 
                                                             properties = properties))
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in cell_blob_props.iterrows()]

    
    
    cell_blob_props = cell_blob_props[cell_blob_props['area'] > area_thresh_min]
    cell_blob_props = cell_blob_props[cell_blob_props['area'] < area_thresh_max]
    cell_blob_props = cell_blob_props[cell_blob_props['eccentricity'] < eccent_thresh]
    
    blob_count = len(cell_blob_props)
    
    
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in cell_blob_props.iterrows()]
    
    if blob_count > 0:
        
        fig, axes = plt.subplots(ncols=3,nrows=1, figsize=(7, 3), dpi = 200,
                                 sharex=True, sharey=True)
    
        ax = axes.ravel()
    
        ax[0].imshow(img_norm, cmap='hsv')
        #plt.colorbar(ccb, cax=ax[0])
    
        ax[1].imshow(mask)
        for blob in tqdm(blob_coordinates):
            width = blob[3] - blob[1]
            height = blob[2] - blob[0]
            patch = patches.Rectangle((blob[1],blob[0]), width, height, 
                       edgecolor='r', facecolor='none')
            ax[2].add_patch(patch)
        ax[2].imshow(img_norm, cmap='gray');
        #ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
    
    
        fig.tight_layout()
        plt.show()
        
    return mask
    


def main():
    
    global show_plots
    min_track_len = 2
    show_plots = False
    
    directory = sys.argv[1]
    files = filepull(directory)
    sum_files = [file for file in files if ".tif" in file and "SUM" in file]
    tracks_files = [file for file in files if "tracks.csv" in file]
    
    sum_files.sort()
    tracks_files.sort()
    
    
    for ii, file in enumerate(sum_files):
        
        # read the sum img
        sum_img = tif.imread(file)
        # segment the blobs 
        sum_mask, norm_img = logblob(sum_img, plot_fig = show_plots)
        
        
        # read the tracks file
        tracks_df = csv2df(tracks_files[ii])
        n_tracks = total_tracks(tracks_df)
        file_index = [ii for x in range(len(tracks_df.index))]
        
        
        new_mask = check_blobs(norm_img, sum_mask, tracks_df, n_tracks,
                               plot_fig = show_plots )
        
        # for each track
        store_tracks = []
        for ll in range(n_tracks):
            
            # define track
            track = tracks_df.loc[tracks_df.TRACK_N == (ll+1)]
            
            # keep count of locs within blob
            loc_counter = 0
            # for each loc
            for mm, loc in enumerate(range(len(track))):
                
                # coordinate of loc
                x = int(track.iloc[mm, 2])
                y = int(track.iloc[mm, 1])
                
                # if coordinate is in blob increase counter
               
                if new_mask[y][x] != 0:
                    loc_counter += 1
                    
            frac_track_in_blob = loc_counter/len(track)
            if frac_track_in_blob >= 0.99:
                
                blob_track_class = ['in' for nn in range(len(track))]
                track['blob_track_class'] = blob_track_class
                
            elif frac_track_in_blob >= 0.25 and frac_track_in_blob < 0.99:
                
                blob_track_class = ['io' for nn in range(len(track))]
                track['blob_track_class'] = blob_track_class
                
            elif frac_track_in_blob < 0.25:
                
                blob_track_class = ['out' for nn in range(len(track))]
                track['blob_track_class'] = blob_track_class
            
            store_tracks.append(track)
             
        #print(tracks_dict)
        #tracks_df_new = pd.DataFrame.from_dict(tracks_dict, orient='index' )
        #print(tracks_df_new)
        new_tracks_df = pd.concat(store_tracks)
        print(new_tracks_df)
        new_tracks_df.to_csv(tracks_files[ii][:-4] + '_blobs.csv', 
                  index = False)
                
                
            
            
                
                
            
           
        
           
        
    
main()    