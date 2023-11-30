#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:43:07 2023

@author: christopherazaldegui
"""

import sys
import numpy as np
import glob

import tifffile as tif
from PIL import Image
from scipy import ndimage as ndi


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

    
    
    
directory = sys.argv[1]
# pull in all of the fits.mat files
files = filepull(directory)

png_files = [file for file in files if 'cp_masks.png' in file]

print("{} .pngfiles found".format(len(png_files)))
png_files.sort()


f = 1
for ii,file in enumerate(png_files):
    
    
    img = Image.open(file) 

    
    tif.imwrite(file[:-12] + 'mask.tif', img)