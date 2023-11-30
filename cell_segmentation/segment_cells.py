# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 20:17:00 2021

@author: azaldegc

Script to segment bacteriall cells from phase contrast images

python cell_segmentation.py pathtofiles\*.tif

Make sure that the only tif files are the phase constrast images
"""

import sys
import numpy as np
import glob

import tifffile as tif
from PIL import Image
import matplotlib.pyplot as plt
from cellpose import models, plot
from skimage import (filters, morphology, segmentation)
import tifffile as tif


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    
    filenames = [img for img in glob.glob(directory)] # create list of image files in directory
    
    return filenames


def read_TIFFStack(stack_file):
    '''Reads TIFF stacks, returns stack and number of frames'''
    
    # stack_file: TIFF stack file name to read
    #-----------------------------------------

    stack = tif.imread(stack_file) # read TIFF stack into a np array
    n_frames = stack.shape[0] # number of frames in stack
    print(stack.shape)
   
    return stack

def preprocess(img):
    return filters.gaussian(img, 1)

def segment_cells(model, filename, img, user_diam, PlotFig=False):
    
    blur_img = img
  
   
    channels = [0,0]
    masks, flows, styles, = model.eval(blur_img, diameter=user_diam, 
                                              channels=channels)
    
    masks = morphology.remove_small_objects(masks, 200) 
    masks = segmentation.clear_border(masks)
    
    # results for verification, if running through data set as false
    if PlotFig == True:
        
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
        plt.show()

    return masks, blur_img

def save_stack(stack, name):
    '''Save a list of frames as a TIFF stack, returns stack'''
    # stack: list of frame to save as TIFF stack
    # name: filename of original stack
    #-------------------------------------------
    
    print("new shape", stack.shape)
    name = name + '_masks.tif'
    tif.imsave(name, stack)
    print(name + " saved")
    
    
    
directory = sys.argv[1] # foldername that contains all of the fits.mat files
tif_files = filepull(directory) # pull in all of the fits.mat files
print("{} .tif files found".format(len(tif_files)))

#---------------------------------------
plot_fig = False
#name = 'cI_0h_1'

model_ec = models.CellposeModel(gpu=False, pretrained_model='T:\Lab_Members\Chris_Azaldegui\Code\cellpose_models\cellpose_residual_on_style_on_concatenation_off_train_2021_07_22_00_30_46.822532')
model_yh = models.CellposeModel(gpu=False, pretrained_model='T:\Lab_Members\Chris_Azaldegui\Code\cellpose_models\cellpose_residual_on_style_on_concatenation_off_train_2021_10_11_12_47_03.869415')

model = model_yh
#-----------------------------------------

f = 1
for ii,file in enumerate(tif_files):
    print(file)
    mask_stack = []
   # img = Image.open(file)
    mov = read_TIFFStack(file)
    
 
    if len(mov.shape) == 2:# single image
        img_inv = preprocess(mov[1])
  

        segmented_cells, img_process = segment_cells(model, file, img_inv,
                                     user_diam = None)
    
        mask_stack.append(segmented_cells)
    
        #tif.imsave(directory[:-5] + name + '_{num:03d}_PhaseMask.tif'.format(num=f),
        #          segmented_cells)
        
        if plot_fig == True:
            fig, ax = plt.subplots(ncols=3, figsize=(6, 3), 
                                   sharey=True, sharex=True)
            ax[0].imshow(mov, cmap='gray')
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            ax[1].imshow(img_inv, cmap='gray')
            ax[1].contour(segmented_cells, z=-1, levels=np.amax(segmented_cells), 
                          linewidth=0.05, colors='w')
            ax[1].set_title('Pre-processed')
            ax[1].axis('off') 
            ax[2].imshow(segmented_cells, cmap='inferno')
            ax[2].set_title('Segmented cells')
            ax[2].axis('off')
    
            fig.tight_layout()
            plt.show()
        
    if len(mov.shape) == 3: # stack 
        
        pc_img = mov[1]
        fl_img= mov[0]
        
        
        pp_img = preprocess(pc_img)
        segmented_cells, img_process = segment_cells(model, file, pp_img, 
                                                     user_diam=None)
        
        if plot_fig == True:
                fig, ax = plt.subplots(ncols=3, figsize=(6, 3), 
                                   sharey=True, sharex=True)
                ax[0].imshow(pc_img, cmap='gray')
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                ax[1].imshow(pc_img, cmap='gray')
                ax[1].contour(segmented_cells, z=-1, levels=np.amax(segmented_cells), 
                          linewidth=0.05, colors='w')
                ax[1].set_title('Pre-processed')
                ax[1].axis('off') 
                ax[2].imshow(segmented_cells, cmap='inferno')
                ax[2].set_title('Segmented cells')
                ax[2].axis('off')
    
                fig.tight_layout()
                plt.show()
        
    save_stack(np.asarray(segmented_cells), file[:-4])
    
            #tif.imsave(directory[:-5] + name + '_{num:03d}_PhaseMask.tif'.format(num=f),
            #          segmented_cells)
        
            
    
            #print(file, name + '_{num:03d}_PhaseMask.tif'.format(num=f))
    

    f += 1