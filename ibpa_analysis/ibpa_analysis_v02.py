 # -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:22:42 2022

@author: azaldegc


Script to analyze IbpA dataset for collaboration project with Y

"""

import sys
import numpy as np
import pandas as pd
import glob
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import (io, feature, filters, measure, color, 
                     segmentation, restoration)
from tqdm import tqdm
from scipy.optimize import curve_fit

# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def read_TIFFStack(stack_file):
    '''Reads TIFF stacks, returns stack and number of frames'''
    
    # stack_file: TIFF stack file name to read
    #-----------------------------------------

    stack = tif.imread(stack_file) # read TIFF stack into a np array
    n_frames = stack.shape[0] # number of frames in stack
   
    return stack, n_frames


def findblobs(img, norm_thresh=0.25, area_thresh=16, eccent_thresh=0.8, plot=True):
    
    img_norm = img / np.amax(img)
    lower_mask = img_norm > norm_thresh
    upper_mask = img_norm <= 1.0
    mask = upper_mask * lower_mask
    mask = segmentation.clear_border(mask, buffer_size = 15, bgval=0)
    
    blobs, n_blobs = measure.label(mask, background = 0, return_num = True)
    
    properties =['area','bbox','convex_area','bbox_area','centroid',
                 'major_axis_length', 'minor_axis_length',
                 'eccentricity']
   

    blob_props = pd.DataFrame(measure.regionprops_table(blobs, 
                                                         properties = properties)) 

    blob_count = len(blob_props)
    
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in blob_props.iterrows()]
    
    blob_props = blob_props[blob_props['area'] > area_thresh]
    blob_props = blob_props[blob_props['eccentricity'] < eccent_thresh]
    
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in blob_props.iterrows()]
    
    
    
    if plot==False:
        
        fig, axes = plt.subplots(ncols=3,nrows=1, figsize=(7, 3), dpi = 300,
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
        ax[2].imshow(img, cmap='gray');
        #ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
    
    
        fig.tight_layout()
        plt.show()
    
    return blob_props


def gauss2D(x, y, x0, y0, sigma, A, offset):
    '''Take parameters to form 2D Gaussian function, return Gaussian function'''
    # x, y: values of point
    # x0, y0: current point
    # sigma: standard deviation
    # A: amplitude of function
    # offset: shift to avoid negative values
    #----------------------------------------
    
    G = A * np.exp(-1*((x-x0)**2 / (2*sigma**2) + (y-y0)**2 / (2*sigma**2))) + offset
    return G

def _gauss(M, *args):
    '''Callable that is passed to curve_fit, returns array with fit vals'''
    
    # Callable that is passed to curve_fit. M is a (2,N) array where N is the
    # total number of data points in Z, which will be ravelled to 1D.
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//5):
        arr += gauss2D(x, y, *args[i*5:i*5+5])
    return arr

def get_guess_params(data, n_guess):
    '''Determines guess parameters for fitting function'''
    
    # data: data to generate parameters for
    # n_guess: number of parameter sets to generate, ie # of particles to look for
    #----------------------------------------------------------------------------
    
    global est_particle_size
    sum_frame = data.sum()# sum of entire frame intensities
    Y, X = np.indices(data.shape)# indices of frame
    
    x = (X*data).sum()/sum_frame # sum of (x-coord * int) / total int
    y = (Y*data).sum()/sum_frame # sum of (y-coord * int) / total int
    
    # estimated particle size is ~ particle size in pix (dfrlmz) * enlarge scale
    est_particle_size = dfrlmz
    # est size / full width half max
    sigma = est_particle_size / (2*np.sqrt(2*np.log(5)))
    
    amplitude = data.max() # amplitude of data aka the max value
    offset = sum_frame / data.size # average intensity in frame

    # Generate guess parameters for target number of guesses
    all_guess_params = []
    guess_params = [x, y, sigma, amplitude, offset] # list of parameters
    for i in range(n_guess): # generate list of lists if multiple fits
        all_guess_params += guess_params
    all_guess_params = np.array(all_guess_params)
    # If there is a NaN value in guess parameters convert it to 1
    all_guess_params[np.isnan(all_guess_params)] = 1        

    return all_guess_params


def get_param_bounds(data, n_guess):
    '''Determine parameter boundaries for quicker fitting, returns param bounds'''
    # data: array of data
    # n_guess: number of param bound sets to generate, ie # of particles
    #--------------------------------------------------------------------
    
    y0_max, x0_max = np.inf, np.inf # x,y max bound is infinite
    y0_min, x0_min = 0, 0 # x,y min bound is 0, there are no negative intensities
    
    # sigma min is sigma / tolerance, max is sigma * tolerance
    sigma_min = (est_particle_size / (2*np.sqrt(2*np.log(5)))) / stdtol
    sigma_max= (est_particle_size / (2*np.sqrt(2*np.log(5)))) * stdtol
    
    amp_min = data.max() * 0.5 # amp min is 50% of max value
    amp_max = data.max() * 1.5 # amp max is 150% of max value
    
    offset_min = 0 # offset min is 0, or no offset
    offset_max = 1000 # offset max is 1000
    
    min_bounds = x0_min, y0_min, sigma_min, amp_min, offset_min # tuple of min bounds
    max_bounds = x0_max, y0_max, sigma_max, amp_max, offset_max # tuple of max bounds
    
    all_min_bounds = () # initiate sets of min bounds
    all_max_bounds = () # initiate sets of max bounds
    for i in range(n_guess): # make sets for each particle
        all_min_bounds += min_bounds
        all_max_bounds += max_bounds
    set_bounds = (all_min_bounds, all_max_bounds) # tuple of max and min bound sets
    
    return set_bounds

def fit_2D_gauss(roi):
    
    # 2D Gaussian fitting to track bead
    yy, xx = np.mgrid[0:roi.shape[0], 
                      0:roi.shape[1]]
        
    guess_parameters = get_guess_params(roi, 1) # get guess parameters
    parameter_bounds = get_param_bounds(roi, 1) # get param bounds
    
    xdata = np.vstack((xx.ravel(), yy.ravel())) # flatten 2D grid to 1D
    popt, pcov = curve_fit(_gauss, xdata, roi.ravel(), # fit Gaussian
                           guess_parameters, bounds = parameter_bounds, 
                           maxfev = 1000)
    
    return popt
    

def fwhm(sigma):
    
    return (2*(2*np.log(2))**0.5)*sigma


def create_fit_image(img, fit):
    
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    fit_img = np.zeros(img.shape)
    fit_img += gauss2D(xx, yy, *fit)
    
    return fit_img


def foci_profile(mcher_norm_img, ibpa_norm_img):
    
    
    print(mcher_norm_img.shape)
    mcher_profile = mcher_norm_img[15,:] 
    ibpa_profile = ibpa_norm_img[15,:] 
    x = np.asarray([ii*.066 for ii in range(len(mcher_profile))])
    
    mcher_profile_norm = mcher_profile / np.amax(mcher_profile)
    ibpa_profile_norm = ibpa_profile / np.amax(mcher_profile)
    
    fig, axes = plt.subplots(figsize=(6,2.5),nrows=1, ncols=3, 
                                 sharey=False, sharex=False, dpi=300)
    ax = axes.ravel()
    
    ax[0].imshow(mcher_norm_img, cmap='inferno')
    ax[1].imshow(ibpa_norm_img, cmap='viridis')
    ax[2].plot(x,mcher_profile_norm, '-', linewidth=3, color='magenta')
    ax[2].plot(x,ibpa_profile_norm, '-', linewidth=3, color='darkgreen')
    fig.tight_layout()
    plt.show()
    
def foci_profile_all(mcher_signals, ibpa_signals):
    
    mcher_all = []
    ibpa_all = []
    font_size = 12
    for ii in range(len(mcher_signals)):
        
        mcher_profile = mcher_signals[ii][15,:]
        ibpa_profile = ibpa_signals[ii][15,:]
        
        mcher_all.append(mcher_profile) #/ np.amax(mcher_profile) )
        ibpa_all.append(ibpa_profile)# / np.amax(ibpa_profile) )
        
    mcher_arr = np.asarray(mcher_all)
    ibpa_arr = np.asarray(ibpa_all)
    
    mcher_mean = np.mean(mcher_arr, axis = 0)
    ibpa_mean = np.mean(ibpa_arr, axis = 0)
    mcher_std = np.std(mcher_arr, axis = 0)
    ibpa_std = np.std(ibpa_arr, axis = 0)
    
    x = np.asarray([ii*.065 - 1 for ii in range(len(mcher_mean))])
    fig, axes = plt.subplots(figsize=(2.75,2.5),
                                 sharey=False, sharex=False, dpi=300)
    print("mcherry", mcher_mean[15], mcher_std[15])
    print("ibpa", ibpa_mean[15], ibpa_std[15])
    print(ibpa_mean[15] / mcher_mean[15])
    
    
    axes.plot(x,mcher_mean, '--', linewidth=3, color='magenta')
    axes.plot(x,ibpa_mean, '--', linewidth=3, color='darkgreen')
    axes.fill_between(x, mcher_mean - mcher_std, mcher_mean + mcher_std, 
                      color = 'magenta', alpha = 0.5)
    axes.fill_between(x, ibpa_mean - ibpa_std, ibpa_mean + ibpa_std, 
                      color = 'darkgreen', alpha = 0.5)
    
    axes.set_ylim(0,4)
    axes.set_xlim(-1, 1)
    
    #if ii == 1:
    axes.set_xlabel('\u03bcm', fontsize=font_size)
   
    axes.set_ylabel('Normalized Intensity', fontsize=font_size)
    axes.tick_params(axis='x',labelsize=font_size)
    axes.tick_params(axis='y',labelsize=font_size)

    fig.tight_layout()
    plt.show()
    
    print(mcher_arr.shape, ibpa_arr.shape)
    
    
    
    
    
    return 

def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

    ax[0].imshow(image, cmap='hsv')
    ax[0].set_title('Original image')
    ax[0].axis('off')

    ax[1].imshow(background, cmap='hsv')
    ax[1].set_title('Background')
    ax[1].axis('off')

    ax[2].imshow(image - background, cmap='hsv')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()    

def main():
    
    global dfrlmz, stdtol 
    plot_fig = True
    directory = sys.argv[1]
    files = filepull(directory)
    dfrlmz = 14
    stdtol = 1.5
    print(files)
    
    
    ibpa_signals = []
    mcher_signals = []
    for file in files:
        
        
        
        
        # read file
        img_stack, _ = read_TIFFStack(file)
        #print(img_stack.shape)
        
        ibpa_chan = img_stack[0]
        mcher_chan = img_stack[1]
        
        # correct for background
        mcher_bg = restoration.rolling_ball(mcher_chan, radius = 10)
        mcher_bg_corr = mcher_chan - mcher_bg
       # plot_result(mcher_chan, mcher_bg)
       # plt.show()
        
        ibpa_bg = restoration.rolling_ball(ibpa_chan, radius = 10)
        ibpa_bg_corr = ibpa_chan - ibpa_bg
       # plot_result(ibpa_chan, ibpa_bg)
       # plt.show()
        
        
        
        
        
        blob_props = findblobs(mcher_bg_corr, plot=True)
        
        
        
        
        print(len(blob_props))
        for index, row in blob_props.iterrows():
            
            blob_centroid = int(row['centroid-0']), int(row['centroid-1'])
            bbox_rad = int(15)
            
            
            
            ibpa_crop = ibpa_bg_corr[blob_centroid[0] - bbox_rad:blob_centroid[0] + bbox_rad + 1, blob_centroid[1] - bbox_rad:blob_centroid[1] + bbox_rad + 1]
            
            mcher_crop = mcher_bg_corr[blob_centroid[0] - bbox_rad:blob_centroid[0] + bbox_rad + 1, blob_centroid[1] - bbox_rad:blob_centroid[1] + bbox_rad + 1]
            
            
            mcher_norm = (mcher_crop - np.amin(mcher_crop)) / (np.amax(mcher_crop) - np.amin(mcher_crop))
            ibpa_norm = (ibpa_crop - np.amin(ibpa_crop)) / (np.amax(ibpa_crop) - np.amin(ibpa_crop))
            
            #foci_profile(mcher_crop, ibpa_crop)
            
            
            
            ibpa_signals.append(ibpa_norm)
            mcher_signals.append(mcher_norm)
            
            
            #######################################################
            '''
            mcher_fit = fit_2D_gauss(mcher_norm)
            ibpa_fit = fit_2D_gauss(ibpa_norm)
            
            mcher_fit_img = create_fit_image( mcher_norm, mcher_fit)
            ibpa_fit_img = create_fit_image(ibpa_norm, ibpa_fit)
            
            
            
            
            if plot_fig == False:
                fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(6, 3), 
                               sharey=True, sharex=True, dpi=300)
                ax=axes.ravel()
            
                ax[0].imshow(mcher_norm, cmap='inferno')
                cs = ax[0].contour(mcher_fit_img, z=0, levels=3, 
                          linestyles='dashed', linewidth=1, colors='w')
                ax[0].clabel(cs, inline=1, fontsize=10)
                ax[0].set_title('mCherry')
                ax[0].axis('off') 
                ax[1].imshow(ibpa_norm, cmap='viridis')
                ci = ax[1].contour(ibpa_fit_img, z=0, levels=3, 
                          linestyles='dashed', linewidth=1, colors='w')
                ax[1].clabel(ci, inline=1, fontsize=10)
                ax[1].set_title('IbpA-sfEGFP')
                ax[1].axis('off')
            
                fig.tight_layout()
                plt.show()
                
            '''
           
       
    ibpa_all = np.zeros((ibpa_signals[0].shape[0],ibpa_signals[0].shape[1]), 
                            np.float64)
    for signal in ibpa_signals:
            
        ibpa_all += signal
        
    mcher_all = np.zeros((ibpa_signals[0].shape[0],ibpa_signals[0].shape[1]), 
                         np.float64)
    for signal in mcher_signals:
        mcher_all += signal
            
    #ibpa_all = ibpa_all / len(ibpa_signals)
    #mcher_all = mcher_all / len(mcher_signals)
       
        
    '''  
    mcher_fit = fit_2D_gauss(mcher_all)
    ibpa_fit = fit_2D_gauss(ibpa_all)
        
        
    mcher_fit_img = create_fit_image(mcher_all, mcher_fit)
    ibpa_fit_img = create_fit_image(ibpa_all, ibpa_fit)
        #print(mcher_fit)
        #print(ibpa_fit)
        
    mcher_fwhm = fwhm(mcher_fit[2])
    ibpa_fwhm = fwhm(ibpa_fit[2])
    print(mcher_fwhm, ibpa_fwhm, ibpa_fwhm / mcher_fwhm, len(mcher_signals))
    '''
        
    if plot_fig == True:
            fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(6, 3), 
                               sharey=True, sharex=True, dpi=300)
            ax=axes.ravel()
            
            im1 = ax[0].imshow(mcher_all, cmap='inferno', vmin=0)
            #cs = ax[0].contour(mcher_fit_img, z=0, levels=[0.25, 0.5, 0.75], 
             #             linestyles='dashed', linewidth=1, colors='k')
            #ax[0].clabel(cs, inline=1, inline_spacing = 10, fontsize=9)
            plt.colorbar(im1, ax=ax[0])
            ax[0].set_title('mCherry')
            ax[0].axis('off') 
            im2 = ax[1].imshow(ibpa_all, cmap='viridis', vmin=0)
            #ci = ax[1].contour(ibpa_fit_img, z=0, levels=[0.25, 0.5, 0.75],
             #             linestyles='dashed', linewidth=1, colors='k')
            #ax[1].clabel(ci, inline=1, fontsize=9, inline_spacing = 10)
            plt.colorbar(im2, ax=ax[1])
            ax[1].set_title('IbpA-sfEGFP')
            ax[1].axis('off')
            
            fig.tight_layout()
            plt.show()
    
    
   
    foci_profile_all(mcher_signals, ibpa_signals)
    
        
    print("total foci analyzed:", len(mcher_signals))
    '''
    fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(6, 3), 
                               sharey=True, sharex=True, dpi=300)
    ax=axes.ravel()
    
    ax[0].imshow(mcher_all, cmap='inferno', vmin=0, vmax=1)
    cs = ax[0].contour(mcher_fit_img, z=0, levels=[0.25, 0.5, 0.75], 
                       linestyles='dashed', linewidth=1, colors='k')
    ax[0].clabel(cs, inline=1, inline_spacing = 10, fontsize=9)
    ax[0].set_title('mCherry')
    ax[0].axis('off') 
    ax[1].imshow(ibpa_all, cmap='viridis', vmin=0, vmax=1)
   # ci = ax[1].contour(ibpa_fit_img, z=0, levels=[0.25, 0.5, 0.75],
    #                   linestyles='dashed', linewidth=1, colors='k')
    #ax[1].clabel(ci, inline=1, fontsize=9, inline_spacing = 10)
    ax[1].set_title('IbpA-sfEGFP')
    ax[1].axis('off')
            
    fig.tight_layout()
    plt.show()
    '''
   
main()