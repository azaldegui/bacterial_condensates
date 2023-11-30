# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:12:44 2023

@author: azaldegc
"""


import sys
import numpy as np
import pandas as pd
import glob
import tifffile as tif
import statistics as ss
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.patches as patches

from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)




# Pull files from directory
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
  #  n_frames = stack.shape[0] # number of frames in stack
   # print(stack.shape)
   
    return stack


def findblobs(cell,cellmask, norm_thresh = 0.65, area_thresh = 10, 
              eccent_thresh = 0.65, plot=True):
    '''
    Approach from 
    https://towardsdatascience.com/image-processing-with-python-blob-detection-using-scikit-image-5df9a8380ade

    Parameters
    ----------
    cell : TYPE
        DESCRIPTION.
    cellreal : TYPE
        DESCRIPTION.
    norm_thresh : TYPE, optional
        DESCRIPTION. The default is 0.8.
    area_thresh : TYPE, optional
        DESCRIPTION. The default is 7.
    eccent_thresh : TYPE, optional
        DESCRIPTION. The default is 0.65.
    plot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    blob_count : TYPE
        DESCRIPTION.

    '''
    
    # create mask to find the blob
    cell_blur = cell*cellmask#filters.gaussian(cell, 0.5)
    #cell_blur = cell
    cell_norm = (cell_blur - np.amin(cell_blur)) / (np.amax(cell_blur) - np.amin(cell_blur))
    #cell_norm  = cell_blur / np.amax(cell_blur)
    
    lower_mask = cell_norm > norm_thresh
    upper_mask = cell_norm <= 1.0
    mask = upper_mask * lower_mask
    mask = mask * cellmask
    
    filtered_mask = mask.copy()
    # filter through the blobs 
    cell_blobs, n_cell_blobs = measure.label(filtered_mask, 
                                             background = 0,return_num = True)
    regions = measure.regionprops(cell_blobs)
   
    properties =['area','bbox','convex_area','bbox_area',
                 'major_axis_length', 'minor_axis_length',
                 'eccentricity', 'label']
    
    

    for reg in regions:
        if reg.area < area_thresh:
            for coord in reg.coords:
                filtered_mask[coord[0]][coord[1]] = 0
        elif reg.eccentricity > eccent_thresh:
            for coord in reg.coords:
                filtered_mask[coord[0]][coord[1]] = 0
        else:
            for coord in reg.coords:
                filtered_mask[coord[0]][coord[1]] = reg.label
    
    filtered_mask = segmentation.clear_border(filtered_mask)
    
    cell_blobs, n_cell_blobs = measure.label(filtered_mask, 
                                             background = 0,return_num = True)
            
    cell_blob_props = pd.DataFrame(measure.regionprops_table(cell_blobs, 
                                                             properties = properties))

        
    
    cell_blob_props = cell_blob_props[cell_blob_props['area'] > area_thresh]
    cell_blob_props = cell_blob_props[cell_blob_props['eccentricity'] < eccent_thresh]
 
    
    blob_count = len(cell_blob_props)
 
        
    
    
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in cell_blob_props.iterrows()]
    cyto_mask = cellmask * np.logical_not(filtered_mask).astype(int)
    if plot==True:
        #print(cell_blob_props)
        fig, axes = plt.subplots(ncols=6,nrows=1, figsize=(7, 3), dpi = 200,
                                 sharex=True, sharey=True)
    
        ax = axes.ravel()
    
        ax[0].imshow(cell_norm, cmap='hsv') # normalied fluorescence image
        #plt.colorbar(ccb, cax=ax[0])
        ax[1].imshow(cellmask) # mask of cell
        ax[2].imshow(mask) # mask of blob
        ax[3].imshow(filtered_mask)
        ax[4].imshow(cyto_mask)
        for blob in tqdm(blob_coordinates):
            width = blob[3] - blob[1]
            height = blob[2] - blob[0]
            patch = patches.Rectangle((blob[1],blob[0]), width, height, 
                       edgecolor='r', facecolor='none')
            ax[5].add_patch(patch)
        ax[5].imshow(cell, cmap='gray');
        #ax[0].set_axis_off()
        for ll in range(1,6):
            ax[ll].set_axis_off()
       
    
    
        fig.tight_layout()
        plt.show()
    
    
    return blob_count, filtered_mask, cyto_mask



def focus_detect(mask, img, file_id, label, rep_no, t, plotfig=False):
    
    labels, n_labels = measure.label(mask, background = 0,return_num = True)
    cells = measure.regionprops(labels, intensity_image=img)  
    data = []
    cell_id = []
    #print("number of regions to start: {}".format(len(cells)))
    for kk, cell in enumerate(cells):
        
        # only look at regions that resemble the width of a cell to 
        # avoid any missegmented region
        cell_width = cell.minor_axis_length
        cell_length = cell.major_axis_length
       
        if cell_width <= 16 and cell_width >= 12 and cell_length >= 25: 
            # assign cell ID to be the same label of the mask
            cell_label = mask[int(cell.centroid[0]), int(cell.centroid[1])]
            #print("cell label = ", cell_label, cell.label)
           # print("cell label",cell_label)
            # get mask region, but exclude surrounding cells in the roi
            ex = 1 
          
            cell_mask_og = mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
          
            cell_mask_og[cell_mask_og != cell_label] = 0   
            cell_mask_og[cell_mask_og == cell_label] = 1
            mask_hull = morphology.convex_hull_image(cell_mask_og)
            
            
        
            # only look at cells that have a non zero label
            #### try to look into why some cells get a zero label!!!!!
            if cell_label > 1:# and len(cell_mask_og) > 0:
                
                # store the cell label
                cell_id.append(cell_label)
                
              
                # select the cell region for the fluorescence channel
                # combine the convex hull mask and the fluorescence channel roi
                
                cell_region = img[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
                cellfluor = cell_region * cell_mask_og#mask_hull
                
                # gather all of the pixel intensities within the cell

                
          
                    
                 
                # classify cell as if with a focus or not
                cell.image[cell.image > 0] = 1
               
                cell_fluor = img[cell.bbox[0]-ex:cell.bbox[2]+ex,
                                 cell.bbox[1]-ex:cell.bbox[3]+ex] 
                
                nblobs, blob_mask, cyto_mask = findblobs(cell_fluor, 
                                   mask_hull, norm_thresh = 0.7,
                                   eccent_thresh = 0.8, plot=False)
             
                
                
                data.append((file_id, cell_label, label, rep_no,t, nblobs))
        
                # plot cell by cell 
                if plotfig == True:
                    fig, axes = plt.subplots(figsize=(3,3))
                    
                    axes.set_title('cell #: {}'.format(cell_label))
                    axes.imshow(cellfluor, cmap='inferno')
                    axes.contour(cell_mask_og, levels=0, colors='w', 
                                  linestyle='dashed')
                    axes.get_xaxis().set_visible(False)
                    axes.get_yaxis().set_visible(False)
                
                   
                    fig.tight_layout()
                    plt.show()

    return data

label = 'McdB'
replicate_no = 3
time = 5
font_size = 16



directory = sys.argv[1]
files = filepull(directory)
phasemask_files = [file for file in files if "_masks.tif" in file]
fluor_files = [file for file in files if "_masks" not in file]
#phasemask_files.sort()
#fluor_files.sort()
print(phasemask_files)
print(fluor_files)

data = [[] for i in range(6)]
for ii, file in enumerate(fluor_files[:]):
    print(file,phasemask_files[ii])
    fl_img = read_TIFFStack(file)[0]
    mask = read_TIFFStack(phasemask_files[ii])
    
    out = focus_detect(mask, fl_img, ii+1, label, replicate_no, time, plotfig=False)
    
    for oo in out:
        for jj, res in enumerate(oo):
            data[jj].append(res)
    
df = pd.DataFrame(data).transpose()
df.columns = ['FILE_ID', 'CELL_ID',  
              'Label', 'Replicate','Time', "Focus"]
print(df)
df.to_csv(directory[:-5] + label +'_'+ str(time) +  '_' + str(replicate_no) + '_focusdetect.csv', index = False)   