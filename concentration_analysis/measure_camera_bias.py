# -*- coding: utf-8 -*-
"""
@author: azaldegc

Script to build an average bias frame from a 100 frame series captured 
with an EMCCD with 0 ms exposure time and a closed shutter.

"""

import tifffile as tif
import sys
import glob
import numpy as np

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
 
    return stack

# load files
directory = sys.argv[1]
files = filepull(directory)

# organize files
for file in files:
    stack = read_TIFFStack(file)

    crop_stack = stack[:,206:306,206:306]
    
    avg_stack = np.zeros((100,100))
    
    for im in crop_stack[:]:
        
        avg_stack += im 
    
    avg_stack = avg_stack / crop_stack[1:].shape[0]
    tif.imsave(directory[:-8] + 'master_bias_frame.tif', avg_stack)
  
        

