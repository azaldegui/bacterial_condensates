# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:03:39 2023

@author: azaldegc
"""

import sys
import numpy as np
import glob

import tifffile as tif
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from cellpose import models, io, plot
#from cellpose_omni import MODEL_NAMES
from cellpose_omni import models, core, plot
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def segment_cells(model, filename, img, user_diam, PlotFig=False):
    
    blur_img = filters.gaussian(img, sigma=1)# blur image before cellpose
  
    #channels = [0,0]
    chans = [0,0]
    
    mask_threshold = 2.5
    verbose = 0 # turn on if you want to see more output 
    use_gpu = False #defined above
    transparency = True # transparency in flow output
    rescale=None # give this a number if you need to upscale or downscale your images
    omni = True # we can turn off Omnipose mask reconstruction, not advised 
    flow_threshold = 0 # default is .4, but only needed if there are spurious masks to clean up; slows down output
    resample = True #whether or not to run dynamics on rescaled grid or original grid 
    cluster=True # use DBSCAN clustering
    
    
    masks, flows, styles = model.eval(blur_img, channels=chans,rescale=rescale,mask_threshold=mask_threshold,
                                  transparency=transparency,flow_threshold=flow_threshold,omni=omni,
                                  cluster=cluster, resample=resample,verbose=verbose)
    
   # io.masks_flows_to_seg(blur_img, masks, flows, filename, chans)
    
    masks = morphology.remove_small_objects(masks, 200) 
    masks = segmentation.clear_border(masks)
    # results for verification, if running through data set as false
    if PlotFig == True:
        
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=chans)
        plt.tight_layout()
        plt.show()

    return masks, blur_img

# foldername that contains all of the fits.mat files
directory = sys.argv[1]#'/Volumes/jsbiteen/Lab_Members/Chris_Azaldegui/Project_Hn/SPT/2023/2023-04-12/*.tif'
# pull in all of the fits.mat files
files = filepull(directory)

tif_files = [file for file in files if '.tif' in file and "mask.tif" not in file]
#print(tif_files)
print("{} .tif files found".format(len(tif_files)))
tif_files.sort()
#---------------------------------------
plot_fig = False
model_name = 'bact_phase_omni'
model = models.CellposeModel(gpu=False, model_type=model_name)

#model_nuc = models.Cellpose(gpu=False, model_type='nuclei')
#model_ec = models.CellposeModel(gpu=False, pretrained_model='T:\Lab_Members\Chris_Azaldegui\Code\cellpose_models\cellpose_residual_on_style_on_concatenation_off_train_2021_07_22_00_30_46.822532')
#model_yh = models.CellposeModel(gpu=False, pretrained_model='T:\Lab_Members\Chris_Azaldegui\Code\cellpose_models\cellpose_residual_on_style_on_concatenation_off_train_2021_08_06_00_25_04.035389')

#model = model_nuc
#-----------------------------------------

f = 1
for ii,file in enumerate(tif_files):
    
    
    img = tif.imread(file)
    
    img = img
    img_inv = np.invert(img)
    img_inv = np.invert(img_inv)
  
    segmented_cells, img_process = segment_cells(model, file, img_inv,
                                     user_diam = None)
    
    tif.imwrite(file[:-4] + '_mask.tif',
              segmented_cells)
    
    if plot_fig == True:
        fig, ax = plt.subplots(ncols=3, figsize=(6, 3), 
                               sharey=True, sharex=True)
        ax[0].imshow(img, cmap='gray')
        ax[0].contour(segmented_cells, z=0, levels=1, linewidth=0.05, colors='r')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(img_process, cmap='gray')
        ax[1].contour(segmented_cells, z=0, levels=1, linewidth=0.05, colors='r')
        ax[1].set_title('Pre-processed')
        ax[1].axis('off') 
        ax[2].imshow(segmented_cells, cmap='turbo')
        ax[2].set_title('Segmented cells')
        ax[2].axis('off')
    
        fig.tight_layout()
        plt.show()
    
    print(file[:-4] + '_PhaseMask.tif')
    f += 1
    