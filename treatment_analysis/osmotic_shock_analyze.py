# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:16:40 2023

@author: azaldec
"""

import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import scipy.stats as ss
import json
import tifffile as tif
from tqdm import tqdm
import matplotlib.patches as patches
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)



def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def findblobs(cell,cellmask, cell_label, norm_thresh = 0.75, area_lower_thresh = 6, 
              area_upper_thresh = 150,
              eccent_thresh = 0.8, solidity_thresh = 0.7, plot=True):
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
    cell_blur = filters.gaussian(cell, 1)
    #cell_blur = cell
    cell_norm = (cell_blur - np.amin(cell_blur)) / (np.amax(cell_blur) - np.amin(cell_blur))
    cell_norm = cell_norm * cellmask
    cell_norm  = cell_norm / np.amax(cell_norm)
    
    lower_mask = cell_norm > norm_thresh
    upper_mask = cell_norm <= 1.0
    mask = upper_mask * lower_mask
    mask = mask * cellmask
    
    filter_one_mask = mask.copy()
    # filter through the blobs 
    cell_blobs, n_cell_blobs = measure.label(filter_one_mask,  
                                             background = 0,return_num = True)
    regions = measure.regionprops(cell_blobs, intensity_image = cell)
   
    properties =['area','bbox','convex_area','bbox_area',
                 'major_axis_length', 'minor_axis_length',
                 'eccentricity', 'intensity_mean', 'label', 'solidity']
    
    #print(n_cell_blobs)
    
    # first filter : size of regions
    for reg in regions:
        
        if reg.area < area_lower_thresh or reg.area > area_upper_thresh:
            for coord in reg.coords:
                filter_one_mask[coord[0]][coord[1]] = 0
        elif reg.eccentricity > eccent_thresh:
            for coord in reg.coords:
                filter_one_mask[coord[0]][coord[1]] = 0
        
        elif reg.solidity < solidity_thresh:
            for coord in reg.coords:
                filter_one_mask[coord[0]][coord[1]] = 0
        else:
            for coord in reg.coords:
                filter_one_mask[coord[0]][coord[1]] = reg.label
    
    filter_one_mask = segmentation.clear_border(filter_one_mask)
    
    cell_blobs, n_cell_blobs = measure.label(filter_one_mask, 
                                             background = 0,return_num = True)
            
    cell_blob_props = pd.DataFrame(measure.regionprops_table(cell_blobs, 
                                                             intensity_image = cell,
                                                             properties = properties))
    
    
    # find non focus region mean intensity to check for false positives
   # invert_mask_one = ~ filter_one_mask
    filter_two_mask = filter_one_mask.copy()
    filter_one_mask[filter_one_mask == 1] = False
    filter_one_mask[filter_one_mask == 0] = True
    cytoplasm = measure.label(filter_one_mask*cellmask, background = 0)
    cytoplasm_props = measure.regionprops(cytoplasm, intensity_image = cell)
    
    # second filter : intensity of regions
    cell_blobs, n_cell_blobs = measure.label(filter_two_mask,  
                                             background = 0,return_num = True)
    filtered_regs = measure.regionprops(cell_blobs, intensity_image = cell)
    #print(n_cell_blobs)
    for reg in filtered_regs: 
        if reg.intensity_mean < cytoplasm_props[0].intensity_mean * 1.25:
            for coord in reg.coords:
                filter_two_mask[coord[0]][coord[1]] = 0
        else:
            for coord in reg.coords:
                filter_two_mask[coord[0]][coord[1]] = 1
            
        
    
   
 
    
    # final blob_count 
    cell_blobs, final_blob_count = measure.label(filter_two_mask, 
                                             background = 0,return_num = True)
    cell_blob_props = pd.DataFrame(measure.regionprops_table(cell_blobs, 
                                                             intensity_image = cell,
                                                            properties = properties))
    #print(final_blob_count)
 
    
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] )for 
                    index, row in cell_blob_props.iterrows()]
    cyto_mask = cellmask * np.logical_not(filter_two_mask).astype(int)
    if plot==True:
        #print(cell_blob_props)
        fig, axes = plt.subplots(ncols=3,nrows=2, figsize=(7, 5), dpi = 150,
                                 sharex=True, sharey=True)
    
        ax = axes.ravel()
        ax[0].set_title("cell_{}".format(cell_label))
        ax[0].imshow(cell_norm, cmap='hsv') # normalied fluorescence image
        #plt.colorbar(ccb, cax=ax[0])
        ax[1].set_title("cell mask")
        ax[1].imshow(cellmask) # mask of cell
        ax[2].set_title("initial blob mask")
        ax[2].imshow(mask) # mask of blob
        ax[3].set_title("final blob mask")
        ax[3].imshow(filter_two_mask)
        ax[4].set_title("cyto mask")
        ax[4].imshow(cyto_mask)
        for blob in tqdm(blob_coordinates):
            width = blob[3] - blob[1]
            height = blob[2] - blob[0]
            patch = patches.Rectangle((blob[1],blob[0]), width, height, 
                       edgecolor='r', facecolor='none')
            ax[5].add_patch(patch)
        ax[5].set_title("cell blobs")
        ax[5].imshow(cell, cmap='gray');
        #ax[0].set_axis_off()
        for ll in range(1,6):
            ax[ll].set_axis_off()
       
    
    
        fig.tight_layout()
        plt.show()
    
    
    return final_blob_count, filter_two_mask, cyto_mask


def get_region_props(mask, image):
    
    regions, n_regions = measure.label(mask, background = 0,return_num = True)
    
    region_props = measure.regionprops(regions, intensity_image = image)
    
    area = []
    intensity = []
    
    for region in region_props: 
    
        pixel_intensities = []
    
        for coord in region.coords:
            
            pixel_intensities.append(image[coord[0], coord[1]])
            
        region_intensity = sum(pixel_intensities)
        area.append(region.area)
        intensity.append(region_intensity)
    
    return intensity[0], area[0]

def focus_detect(mask, img, file_id, label, plotfig=False):
    

    
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
       
        if cell_width <= 20 and cell_width >= 10 and cell_length >= 20: 
            # assign cell ID to be the same label of the mask
            cell_label = mask[int(cell.centroid[0]), int(cell.centroid[1])]
            
            # get mask region, but exclude surrounding cells in the roi
            ex = 1 
            temp_mask = mask.copy()
            cell_mask_og = temp_mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
          
            cell_mask_og[cell_mask_og != cell_label] = 0   
            cell_mask_og[cell_mask_og == cell_label] = 1
            
            # convex hull the mask if samples is cI_agg
            if label == 'cI_agg':
                mask_hull = morphology.convex_hull_image(cell_mask_og)
            else: 
                mask_hull = cell_mask_og
            
            
        
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
                                   mask_hull, cell_label, norm_thresh = 0.80,
                                   eccent_thresh = 0.85, plot=False)
                
                
                # determine cellular, cytoplasmic, and blob intensities
                
                if nblobs > 0:
                    
                    # get cell intensity, cyto intensity 
                    cell_intensity, cell_area = get_region_props(mask_hull, cell_fluor)
                    cyto_intensity, cyto_area = get_region_props(cyto_mask, cell_fluor)
                    focus_intensity, focus_area = get_region_props(blob_mask, cell_fluor)
                    
                elif nblobs == 0:
                    
                    # get cell intensity, cyto intensity 
                    cell_intensity, cell_area = get_region_props(mask_hull, cell_fluor)
                    cyto_intensity, cyto_area = get_region_props(cyto_mask, cell_fluor)
                    focus_intensity, focus_area = [0], [0] 
                    
               # print(cell_intensity, cyto_intensity, focus_intensity)
             
                
                # store data
                data.append([file_id, cell_label, label, rep,
                             nblobs, 
                             cell_intensity, cell_area, 
                             cyto_intensity, cyto_area
                             ])
        
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


global label, rep
protein = 'PopTag_SL'
label = 'wash_30min'
rep = 3
font_size = 16

# salt, wash_15min, ctrl_15min, ctrl_0min


directory = sys.argv[1]
files = filepull(directory)

# for ms datasets
phasemask_files = [file for file in files if "_mask" in file]
fluor_files = [file for file in files if "_fluor" in file]



phasemask_files.sort()
fluor_files.sort()




for ii, file in enumerate(fluor_files[:]):
    
    # read files
    print(file,phasemask_files[ii])
    img = tif.imread(file)
    mask = tif.imread(phasemask_files[ii])
    
    out = focus_detect(mask, img, ii, label, 
                       plotfig=False)
    
    
    df = pd.DataFrame(out)
    print(df)

    df.columns = ['FILE_ID', 'CELL_ID',  
              'Label', 'Replicate', 
              "Focus", "Cell Intensity", "Cell Area",
              "Cytoplasm Intensity", "Cytoplasm Area"]    
    

    df.to_csv(directory[:-5] + protein + '_' + label + '_' + str(rep) +
              '-' + str(ii) + '_focusdetect.csv', index = False)   


