# -*- coding: utf-8 -*-
"""
@author: azaldegc

Script that takes a .mat file (originally an .nd2 movie) and averages the n
brightest frames in the movie. The average frame generated is used to estimate
the cellular concentration of a protein. 

run script as:
    python mov_mat2tif_bright_avg.py path/to/files/*.mat

"""
import h5py
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import tifffile as tif

# user-defined parameters
n_frames = 5 # number of frames to average

# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# load files
directory = sys.argv[1]
mat_files = filepull(directory)

# integration function over one 2D frame or array
def sum1(input):
    return sum(map(sum, input))

# get average image over the n brightest frames in a 3D array or stack
def brightframes_avg(arr, n, plotfig=False):
     
    # determine the indeces for the largest integrated intensity frames
    ind = np.argpartition(arr.sum(axis=(1,2)),-n)[-n:]
    # print([(arr.sum(axis=(1,2))[x],x) for x in ind])
    
    # make a substack of the frames corresponding to the indices previously extracted
    substack = np.asarray([arr[ii] for ii in ind])
    # initiate 2D zeroes array of the size of single frame
    avg = np.zeros((substack.shape[1],substack.shape[2]), 
                            np.float64)
    # get average of n length substack
    for frame in substack:
        avg += frame
    avg = (avg / n).transpose()
    
    # can plot to confirm output looks correct
    if plotfig == True:
        plt.figure()
        plt.imshow(avg)
        plt.show()

    # return indices, the substack of bright frames, and the average of the substack
    return ind, substack, avg

# wrap code
for file in mat_files:
    
    # read mat file
    with h5py.File(file, 'r') as f:
        
        print(file[:-4])
        # conver mat file to array
        arr = np.asarray(f['mov'])

        # find the indeces for the n frames of which the total sum is largest
        bright_ind, bright_stack, bright_avg = brightframes_avg(arr, n_frames, 
                                                                plotfig=False)
        # save avg image as tif
        tif.imsave(file[:-4] +'_max.tif', bright_avg)