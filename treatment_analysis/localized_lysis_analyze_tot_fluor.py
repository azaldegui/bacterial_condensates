# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 12:45:54 2023

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


# foldername that contains all of the fits.mat files
directory = sys.argv[1]#'/Volumes/jsbiteen/Lab_Members/Chris_Azaldegui/Project_Hn/SPT/2023/2023-04-12/*.tif'
# pull in all of the fits.mat files
files = filepull(directory)

image_files = [file for file in files if '.tif' in file and 'mask.tif' not in file]
mask_files = [file for file in files if 'mask.tif' in file]
 
#print(tif_files)
print("{} .tif files found".format(len(image_files)))
image_files.sort()
mask_files.sort()
#---------------------------------------





#-----------------------------------------

data = []
data_norm = []
font_size = 13
sample = 'McdB_03'
protein = 'McdB'
frame_jump = 50 # in seconds
n_bins = 20
plot_fig = False


store_data = []

f = 1
for ii,file in enumerate(image_files[:]):
    
    
    image = tif.imread(file)
    mask = tif.imread(mask_files[ii])
    
    
   # phase_img = image[0]
    fluor_img = image
    
   
    blob_bool = True
    cell_data = []
    avg_fluor = []
    
        
    labels, n_labels = measure.label(mask, background = 0, return_num = True)
    cells = measure.regionprops(labels)
        
    
    t = ii*frame_jump
        
    for kk, cell in enumerate(cells[:]):
            
        
            
        temp_mask = mask.copy() # use temporary mask to prevent issues
    
        cell_origin = cell.centroid # define the centroid of the cell
        cell_label = mask[int(cell_origin[0]), int(cell_origin[1])] # deterine the label of cell 
            
        ex = 2
        cell_mask = temp_mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the mask to the specific cell
        cell_mask[cell_mask != cell_label] = 0 # remove any other cells in the ROI
        cell_mask[cell_mask == cell_label] = 1 # change cell value to 1 
            
    
        cell_fluor = fluor_img[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the cb image as before
        
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
            
        store_data.append((protein, sample, t, cell_label, condensation_coeff[0], 
                           condensation_coeff[1], condensation_coeff[2],
                           condensation_coeff[3], condensation_coeff[4], 
                           sum(cellpixvals), sum(cellpixvals) / len(cellpixvals)
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
df.columns = ['Protein', 'Sample', 'Time', 'Cell Label', 'I = 0.1',
              'I = 0.3', 'I = 0.5', 'I = 0.7', 'I = 0.9', 'Total Cell Intensity',
              'Average Cell Intensity']

print(df)
df.to_csv(directory[:-5] + sample + '_A22_condensationcoeff.csv', index = False)
            
    
            
  


