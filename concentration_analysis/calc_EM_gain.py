# -*- coding: utf-8 -*-
"""
@author: azaldegc

Script to determine the electron multiplication factor when calibrating an 
EMCCD camera. 

Only use directory that contains the bias frame and the image captured at 
different electron multiplication inputs. 

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
em_files = [file for file in files if "master_bias" not in file]
em_files.sort()

# read bias frame
im_bias = read_TIFFStack(bias_files[0])

# read image without any EM 
no_gain_im = read_TIFFStack(em_files[0])[206:306,206:306]


# define the exposure time for the images captures (in miliseconds)
long_exp = 1000 
short_exp = 10

mean_signals = []
for file in em_files[1:]:
    stack = read_TIFFStack(file)
    im = stack[206:306,206:306] # crop to make center 100 x 100 pix region
   
    # subtract bias from both images and correct for exposure time
    no_gain_bc = (no_gain_im - im_bias) / long_exp
    im_bc = (im - im_bias) / short_exp
    
    EM_factor = np.mean(im_bc) / np.mean(no_gain_bc) 
    print(EM_factor)
    
