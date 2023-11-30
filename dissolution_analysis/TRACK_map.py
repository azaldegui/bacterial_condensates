# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:13:47 2022

@author: azaldegc


Script make single cell trajectory images:
    
    Needs a directory with the tracks.csv and original phase image .tif 
    and phase mask .tif file
    
"""


# import modules
import sys
import numpy as np
import glob
from PIL import Image
import tifffile as tif

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

from matplotlib import cm
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
    df = pd.read_csv(file, header = 0)
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

    return tracks_dict


# calculate MSD curves and apparent diffusion coefficient
def calc_MSD(track, tracks_dict, req_locs, plot=False):
    # corresponding column number for x and y coordinates    
    x_coord = 2
    y_coord = 1
    # determine # of localizations of track
    track_n_locs = len(tracks_dict[track])
    # determine # of steps of track
    track_n_steps = track_n_locs - 1 #req_steps
    # hold list for MSD of a track
    MSD_track = []        
    # start time_lag
    time_lag = 0
    # run a lag calc until max step num
    for lag in range(max_tau): # changed the -1
        # keep count of time lag
        time_lag += 1
        # num of steps to calc
        n_steps = track_n_steps - time_lag
        #global displacement_list
        sq_displacement_list =[]
        # for each step of time lag
        for j, step in enumerate(range(n_steps)):
            x0, xF = loc_in_dict(tracks_dict, track, j, x_coord,time_lag)
            y0, yF = loc_in_dict(tracks_dict, track, j, y_coord,time_lag)
            sq_displacement = (calc_stepsize(x0,xF,y0,yF))**2
            sq_displacement_list.append(sq_displacement)
        mean_sq_dis = np.average(sq_displacement_list)
        MSD_track.append(mean_sq_dis)
    
    # from MSD curves, estimate diffusion coefficient
    tau_list = [(x*framerate) + framerate for x in range(len(MSD_track))]  
    ydata = np.asarray(MSD_track[:5])
    xdata = np.asarray(tau_list[:5])
    diff, _ = curve_fit(brownian_blur_corr, xdata, ydata,
                            maxfev = 10000)
    corr_matrix = np.corrcoef(ydata, brownian_blur_corr(xdata,*diff))
    corr_yx = corr_matrix[0,1]
    r_squared = corr_yx**2
    # return diffusion coefficient
    return diff[0], diff[1], 1, r_squared            

# calculate size of step
def calc_stepsize(x0,xF,y0,yF):
    x_step = xF - x0
    y_step = yF - y0
    stepsize_pix = np.sqrt(x_step**2 + y_step**2)
    stepsize_um = stepsize_pix*pixsize 
    return stepsize_um


# determine position in dictionary
def loc_in_dict(data,key,row,col,time_lag):
    loc0 = data[key].iloc[row,col]
    locF = data[key].iloc[row+time_lag,col]    
    return loc0,locF

def brownian_a(tau,D,a): # normal Brownian motion
    return log(4*D,10) + a*tau

def brownian(tau,D): # normal Brownian motion
    return 4*D*(tau)

def brownian_blur_corr(tau,D, sig): # normal Brownian motion with blurring
    return (8/3)*D*(tau) + 4*sig**2

def main():
    
    # set global parameters
    global pixsize, min_locs, minstepsize, max_tau, framerate
    framerate = 0.04 # frame acquisition every x seconds
    pixsize = 0.049 # microns per pixel 
    minstepsize = 0.00
    min_track_len = 10 # in localizations
    max_tau = 5
    saveplot = True
    showplot = True

    # load fits csvs and phase image tiffs
    directory = sys.argv[1]
    files = filepull(directory)
    phaseImg_files = [file for file in files if ".tif" in file and "PhaseMask" not in file]
    tracks_files = [file for file in files if "tracks.csv" in file]
  #  mask_files = [file for file in files if "PhaseMask.tif" in file]
      
    tracks_files.sort()
    phaseImg_files.sort()
    
    # for each image and fit
    for ii,file in enumerate(tracks_files):
        print(file)
        # read fits csv
        tracks_df = csv2df(file)
        # convert tracks df to dictionary
        dataset = tracks_df2dict(tracks_df, total_tracks(tracks_df), 
                                 min_track_len)
                
        
        
        # read phase img
        phaseImg = tif.imread(phaseImg_files[ii])
    #    phasemask = tif.imread(mask_files[ii])
        blankimg = np.zeros((phaseImg.shape[0],phaseImg.shape[1]), 
                            np.float64)
        phaseImg_blur = filters.gaussian(phaseImg,0.1) 
        # initiate image
        fig,ax = plt.subplots(dpi=300)
      #  plt.rcParams["font.weight"] = "bold"
      #  plt.rcParams["axes.labelweight"] = "bold"
        
        #plt.rcParams['figure.figsize'] = 3,3
  
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)   
        
        
        numrows, numcols = blankimg.shape
        plt.axis('off')
        plt.imshow(blankimg, cmap='gray', zorder=0)
        plt.contour(phaseImg, z=0, levels=4, linewidth=0.001, colors='w')
        #D_range = np.arange(10**-3, 10**0, )
        from matplotlib import colors
        norm = colors.LogNorm(vmin=10**-2, vmax=10**0)
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.hsv)
        cmap.set_array([])
        
        for track in dataset:
            
            D, loc_pres, a, r2 = calc_MSD(track, dataset, min_track_len)
            print(D)
           # if Dand r2 >= 0.5:
           #     print(D) 
            if D > 0:
                sc = ax.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                        c=cmap.to_rgba(D),
                        linewidth=0.75, alpha=0.75)
            elif D < 0 and D > -0.1:
                sc = ax.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                        c='yellow',
                        linewidth=0.75, alpha=0.75)
            else:   
                sc = ax.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                        c='darkorange',
                        linewidth=0.75, alpha=0.75)
            '''
            if D <= 0.01:
                plt.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                       linewidth=1., alpha=0.75, color='green')
            elif D >= 0.01 and D < 0.09:
                plt.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                       linewidth=1., alpha=0.75, color='teal')             
            elif D >= 0.09:
                plt.plot(dataset[track].iloc[:,2], dataset[track].iloc[:,1],
                       linewidth=1., alpha=0.75, color='magenta')
            '''
           
        cbar = fig.colorbar(cmap)
        cbar.ax.set_ylabel('Diffusion coefficient', fontsize=20)
        tick_font_size = 20
        cbar.ax.tick_params(labelsize=tick_font_size)                                                   
        plt.ylim(numrows, 0)
        scalebar = ScaleBar(pixsize, 'um', color = 'k', box_color = 'None',
                            location = 'upper right', fixed_value = 1, scale_loc='none',
                            width_fraction=0.025, label_loc='none', label=None
                            ) 
        #ax.add_artist(scalebar)
        
        fig.tight_layout()
        if saveplot == True:
            plt.savefig(phaseImg_files[ii][:-10]+"trackMap_{num:03d}_v02.png".format(num=ii+1), 
                    dpi=400, 
                    bbox_inches='tight')

        if showplot == True:
            plt.show()
        
main()    