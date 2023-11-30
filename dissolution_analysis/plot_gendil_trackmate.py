# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:21:01 2022

@author: azaldegc

script to plot gen dilution experiments analyzed using TrackMate
- plot the intensity of the cluster over time 
"""


import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
from sklearn.mixture import GaussianMixture
import scipy.stats as stats

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]  
    
    return filenames
    
# Separate tracks by ID into dictionary
def tracks_df2dict(df, n_tracks, min_length):
    
    tracks_dict = {} # initialize empty dictionary
    
    for ii in range(n_tracks): # for each track
        
        track = df.loc[df.TRACK_ID == (ii)] # assign current track to var
        track_length = len(track) # find track length (n localizations)
   
        # if track length > or = min_length then store in dict
        if track_length >= min_length: 
            tracks_dict["track_{0}".format(ii)] = df.loc[df.TRACK_ID == (ii)]

    return tracks_dict

def total_tracks(df):
    n_tracks = df['TRACK_ID'].max()
   
    return int(n_tracks)




def main():
    min_track_len = 1 # minimum track length in frames
    font_size = 13 # fontsize, for plotting
    frame_interval = 15 # in minutes
    n_bins= 10 # number of bins if plotting lifetime and or fluor decay
    color = 'darkslategray' # for plot color
    sample = 'PopTag_LL'
    date = '20220815_'
    stop_frame = 61 #41 for ceph treat, 61 for gendil
    
    # read csv files and convert to dataframes
    directory = sys.argv[1]
    filenames = filepull(directory)
    print(filenames)
    dfs = []
    samples = []
    dicts = []
    
    for file in filenames:
        df = pd.read_csv(file,index_col=0)
        dfs.append(df)
    
    # convert dataframes to dictionary of tracks
    for df in dfs:
        tracks_dict = tracks_df2dict(df, total_tracks(df), min_track_len)
        dicts.append(tracks_dict)
  

    # start plot
    fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(9, 5), 
                             sharey=False, sharex=False, 
                             gridspec_kw={'width_ratios': [3, 1]})
    ax = axes.ravel()
    
    track_lifetimes_frames = []
    track_lifetimes_hrs = []
    fluor_max_diff = []
    fluor_start_diff = []
    ID_list = []
    
    intensity_traces = []
    
    for ll, dataset in enumerate(dicts):
        
        
        for ii,tr in enumerate(dataset):
            track = dataset[tr]
            frames = np.asarray(track['FRAME'])[:stop_frame]
            time = (frames * frame_interval) / 60 # x-axis in hrs
            tot_intensity = np.asarray(track['TOTAL_INTENSITY'])[:stop_frame]
            tot_intensity_norm = tot_intensity / max(tot_intensity) # y-axis normalized total intensity
            
            # only plot tracks that are present a t=0
            if 0 in time:
                ax[0].plot(time, tot_intensity_norm, '-', color=color, 
                         linewidth=.5,alpha=0.5)
                track_lifetimes_frames.append(max(frames)) # get track lifetime
                track_lifetimes_hrs.append(max(time))
                fluor_max_diff.append(tot_intensity_norm[-1] - 
                                     max(tot_intensity_norm[:]))
                fluor_start_diff.append(tot_intensity_norm[-1] - 
                                        tot_intensity_norm[0])
                ID_list.append(str(ll+1) + '_' + tr)
                intensity_traces.append(tot_intensity_norm.tolist())
                
    ax[0].set_xlim(0,15)
    ax[0].set_ylim(0,1.1)
    ax[2].set_xlabel('Time (hrs)', fontsize=font_size)
    ax[0].set_ylabel('Total intensity',fontsize=font_size)
    ax[0].tick_params(axis='x',labelsize=font_size)
    ax[0].tick_params(axis='y',labelsize=font_size)
                                     
        
    sample_list = [sample for jj in range(len(track_lifetimes_frames))]   
    datatosave = [ID_list, sample_list, track_lifetimes_frames,
                  track_lifetimes_hrs, fluor_max_diff, fluor_start_diff]
    dataDF = pd.DataFrame(datatosave).transpose()
    dataDF.columns = ['Track_ID', 'Sample', 'Track Lifetime (frames)', 
                      'Track Lifetime (hrs)',
                      'Fluorescence_max_diff', 'Fluorescence_start_diff']
    
    print(dataDF)
    #print(directory[:-5])
    
         
    
    ax[1].hist(fluor_max_diff, density=True, bins=n_bins,
               histtype = 'bar', color = color,
                       linewidth=1, edgecolor='white')
#    print(np.mean(fluor_decays), len(fluor_decays))
    ax[1].set_xlim(-1, 1)
  
    ax[1].set_xlabel('$Intensity_{f}$ - $Intensity_{max}$', fontsize=font_size)
    ax[1].set_ylabel('Density',fontsize=font_size)
    ax[1].tick_params(axis='x',labelsize=font_size)
    ax[1].tick_params(axis='y',labelsize=font_size)
    
    
    longest_trace = max([len(ii) for ii in intensity_traces])
    norm_int_byindex = [[] for mm in range(longest_trace)]
    time_trace = [(frame * frame_interval)/60 for frame in range(len(norm_int_byindex))]

    for trace in intensity_traces:
        for ii, val in enumerate(trace):
            norm_int_byindex[ii].append(val)
    
    norm_int_mean = np.asarray([np.mean(tt) for tt in norm_int_byindex])
    norm_int_std = np.asarray([np.std(tt) for tt in norm_int_byindex])
    norm_int_n = np.asarray([len(tt) for tt in norm_int_byindex])
    
    avg_save = [time_trace, norm_int_mean, norm_int_std, norm_int_n,
                [sample for hh in range(len(norm_int_mean))]]
    avgDF = pd.DataFrame(avg_save).transpose()
    avgDF.columns = ['time','trace average', 'trace stdev', 'N', 'sample']
    
    ax[2].plot(time_trace, norm_int_mean, color=color)
    ax[2].fill_between(time_trace, norm_int_mean - norm_int_std,
                       norm_int_mean + norm_int_std, alpha=0.15, color=color)
    ax[2].set_xlim(0,15)
    ax[2].set_ylim(0,1.1)
    ax[2].set_ylabel('Total intensity',fontsize=font_size)
    ax[2].tick_params(axis='x',labelsize=font_size)
    ax[2].tick_params(axis='y',labelsize=font_size)
    
    ax[3].hist(fluor_start_diff, density=True, bins=n_bins,
               histtype = 'bar', color = color,
                       linewidth=1, edgecolor='white')
#    print(np.mean(fluor_decays), len(fluor_decays))
    ax[3].set_xlim(-1, 1)
  
    ax[3].set_xlabel('$Intensity_{f}$ - $Intensity_{i}$', fontsize=font_size)
    ax[3].set_ylabel('Density',fontsize=font_size)
    ax[3].tick_params(axis='x',labelsize=font_size)
    ax[3].tick_params(axis='y',labelsize=font_size)
                

    fig.tight_layout()
    plt.show()
    
    dataDF.to_csv(directory[:-5] + date + sample + '_gendil_TrackMate.csv', index=False)
    avgDF.to_csv(directory[:-5] + date + sample + '_gendil_avg_traces.csv', index = False)
        
    
    
main()

