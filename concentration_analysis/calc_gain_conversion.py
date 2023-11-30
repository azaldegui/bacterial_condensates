# -*- coding: utf-8 -*-
"""
@author: azaldegc

Script to determine the conversion gain factor when calibrating an 
EMCCD camera. Adapted from https://www.mirametrics.com/tech_note_ccdgain.php

Only use directory that contains the bias frame and the image captured at 
different exposure times. The analysis done here accounts for flat field effects.

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
bias_files = [file for file in files if "master_bias" in file]
gain_files = [file for file in files if "master_bias" not in file]

# read bias frame
im_bias = read_TIFFStack(bias_files[0])
print(np.std(im_bias)**2)

for file in gain_files:
    stack = read_TIFFStack(file)
      
    # crop to make center 100 x 100 pix region
    crop_stack = stack[:,206:306,206:306]
    
    # assign second frame to A and third frame to B
    # does not matter which frames as long as they are in succesion
    im_a = crop_stack[1]
    im_b = crop_stack[2]
       
    # subtract bias from both images
    im_a_bc = im_a - im_bias
    im_b_bc = im_b - im_bias
    
    # measure the mean signal of each bias corrected image
    S_a = np.mean(im_a_bc)
    S_b = np.mean(im_b_bc)
    
    # calculate the ratio of the mean signals
    r = S_a / S_b
    
    # multiply image b by the scalar r
    im_b_bc_R = im_b_bc * r
    
    # subtract image b from image a
    # this final steps cancels the flat field effects present in both images
    final_im = im_a_bc - im_b_bc_R
    
    # measure the standard deviation over the same pixel region used for S_a
    # square value to get variance and divide by 2.0 to correct for doubling
    # of variance when image is subtracted from a similar one.
    V_a = (np.std(final_im)**2) / 2
    
    # data point for Signal vs. Variance plot
    print(file,"Variance: {}, Signal: {}".format(V_a, np.mean(S_a)))     
    
    
    
  