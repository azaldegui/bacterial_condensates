# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:17:18 2023

@author: azaldec
"""

# phase mask 
import sys
import numpy as np
import glob

import pandas as pd
import tifffile as tif
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from cellpose import models, io, plot
#from cellpose_omni import MODEL_NAMES
from cellpose_omni import models, core, plot
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util, restoration

)


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def segment_cells(model, img, user_diam, PlotFig=False):
    
    blur_img = filters.gaussian(img, sigma=1)# blur image before cellpose
  
    #channels = [0,0]
    chans = [0,0]
    
    mask_threshold = 1.5
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

    masks = morphology.remove_small_objects(masks, 250) 
    masks = segmentation.clear_border(masks)
    # results for verification, if running through data set as false
    if PlotFig == True:
        
        fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=chans)
        plt.tight_layout()
        plt.show()

    return masks, blur_img

def invert_img_stack(stack):
    
    inv_stack = []
    
    for frame in stack:
        inv_stack.append(np.invert(frame))
    return np.array(inv_stack)

# foldername that contains all of the fits.mat files
directory = sys.argv[1]#'/Volumes/jsbiteen/Lab_Members/Chris_Azaldegui/Project_Hn/SPT/2023/2023-04-12/*.tif'
# pull in all of the fits.mat files
files = filepull(directory)

image_files = [file for file in files if 'fluor.tif' in file]
phase_files = [file for file in files if 'phase.tif' in file]
 
#print(tif_files)
print("{} .tif files found".format(len(image_files)))
image_files.sort()
phase_files.sort()
#---------------------------------------


model_name = 'bact_phase_omni'
model = models.CellposeModel(gpu=False, model_type=model_name)


#-----------------------------------------

data = []
data_norm = []
font_size = 13
sample = 'cI_agg'
frame_jump = .0833 # in minutes
n_bins = 20
plot_fig = False


store_data = []

f = 1

# for each movie
for ii,file in enumerate(image_files[:]):
   
    # read movie
    fluor_stack = tif.imread(file)
    phase_stack = tif.imread(phase_files[ii])
    print(file)
    
    
    
    
    # mask phase mask for stack 
    # invert phase stack 
    inverted_phase_stack = invert_img_stack(phase_stack)
    inverted_phase_stack = invert_img_stack(inverted_phase_stack)
    mask_stack = []
    for frame in inverted_phase_stack:
        
        segmented_cells, img_process = segment_cells(model, frame,
                                                     user_diam = None) 
        # plot_mask(frame, segmented_cells, img_process)
        mask_stack.append(segmented_cells)
    tif.imwrite(file[:-4] + '_mask.tif', mask_stack)
             
    
    # for each frame in the stack 
    for jj, frame in enumerate(fluor_stack):
        
        # define time point
        t = jj * frame_jump
        
        # label the mask to mark single cells
        labels, n_labels = measure.label(mask_stack[jj], background = 0, return_num = True)
        cells = measure.regionprops(labels)
        
        # analyze each cell
        for kk, cell in enumerate(cells[:]):
            
        
            
            temp_mask = mask_stack[jj].copy() # use temporary mask to prevent issues
    
            cell_origin = cell.centroid # define the centroid of the cell
            cell_label = mask_stack[jj][int(cell_origin[0]), int(cell_origin[1])] # deterine the label of cell 
            
            ex = 2
            cell_mask = temp_mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the mask to the specific cell
            cell_mask[cell_mask != cell_label] = 0 # remove any other cells in the ROI
            cell_mask[cell_mask == cell_label] = 1 # change cell value to 1 
            
    
            cell_fluor = frame[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the cb image as before
        
            cell_fluor_mask = cell_fluor * cell_mask 
        
            # gather cell pixel intensities
            cellpixvals = []
            cell_fluor_mask_flat = cell_fluor_mask.flatten()
            for pix in cell_fluor_mask_flat:
                if pix > 0:
                    cellpixvals.append(pix)
                      
            # normalize the pixel intensities and bin 
            cellpixvals_norm = [(pix-min(cellpixvals))/(max(cellpixvals)-min(cellpixvals)) 
                            for pix in cellpixvals]
        
            x, bins = np.histogram(cellpixvals_norm, bins=n_bins)
            global bincenters
            bincenters = 0.5 * (bins[1:] + bins[:-1])
       
            y = x / len(cellpixvals_norm)
        
        
            # calculate condensation coefficient
            condensation_thresh = [0.1, 0.3, 0.5, 0.7, 0.9]
            condensation_coeff = []
            for thresh in condensation_thresh:
            
                condensation_coeff.append((sum(y[:int(n_bins*thresh)])))
            
            store_data.append((sample, file, t, cell_label, condensation_coeff[0], 
                           condensation_coeff[1], condensation_coeff[2],
                           condensation_coeff[3], condensation_coeff[4]
                                                                     ))
            
            # plot cell by cell 
            if plot_fig == True:
                fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(7,3))
                ax = axes.ravel() 
                ax[0].set_title('cell #: {}'.format(cell_label))
                ax[0].imshow(cell_fluor_mask, cmap='inferno')
                # ax[0].contour(cell_fluor, levels=0, colors='w', linestyle='dashed')
                ax[0].get_xaxis().set_visible(False)
                ax[0].get_yaxis().set_visible(False)
        
                ax[1].plot(bincenters, y, 'o-', c='k')
                ax[1].set_xlabel('Normalized intensity', fontsize=12)
                ax[1].set_ylabel('Fraction of pixels', fontsize=12)
                ax[1].set_ylim(0,1)
                ax[1].set_xlim(0,1)
            

                fig.tight_layout()
                plt.show()
            
            
df = pd.DataFrame(store_data)
df.columns = ['Sample', 'File', 'Time (min)', 'Cell Label', 'I = 0.1',
              'I = 0.3', 'I = 0.5', 'I = 0.7', 'I = 0.9']

print(df)
df.to_csv(directory[:-5] + sample + '_loclysis_condensationcoeff.csv', index = False)
            