# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:11:10 2023

@author: azaldec
"""

from os.path import join
from pathlib import Path

import nd2
import tifffile as tif
import sys
import glob

folder = r'### some directory ###'
fname = r'### some file name with .nd2 extension ###'



# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


# foldername that contains all of the fits.mat files
directory = sys.argv[1]#'/Volumes/jsbiteen/Lab_Members/Chris_Azaldegui/Project_Hn/SPT/2023/2023-04-12/*.tif'
# pull in all of the fits.mat files
files = filepull(directory)

for file in files:
    image = tif.imread(file)
    
    fluor = image[:,1,:,:]
    phase = image[:,0,:,:]
    
    
    new_fname = file[:-4] + '_fluor.tif'
    new_pname = file[:-4] + '_phase.tif'
    tif.imsave(new_fname, data=fluor)
    tif.imsave(new_pname, data=phase)
    