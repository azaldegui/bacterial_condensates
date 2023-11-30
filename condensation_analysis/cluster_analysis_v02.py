# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:48:36 2021

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

# determine what the background intensity is so that background subtraction can be done
def find_background_int(mask, img, bg='median'):
    
    # invert the mask to target the background, treat background as an region
    inv_mask = np.invert(mask)
    labels, n_labels = measure.label(inv_mask, 
                                     background = 0, return_num=True )
    regions = measure.regionprops(labels)
    
    # get background coordinates
    background_coords = regions[0].coords
    
    
    # determine either the median or average background intensity
    ints = []
    intensity = 0
    for coord in background_coords:
        intensity += img[coord[0]][coord[1]]
        ints.append(img[coord[0]][coord[1]])
    if bg == 'mean':
        bg_intensity = int(intensity / len(background_coords))
    elif bg == 'median':
        bg_intensity = int(ss.median(ints))
    print("bg_intensity", bg_intensity)
    

    return bg_intensity


def gini_coeff(pixels):
    
    # mean absolute difference
    mad = np.abs(np.subtract.outer(pixels, pixels)).mean()
    # relative absolute difference 
    rmad = mad / np.mean(pixels)
    # GINI coefficient
    g = 0.5 * rmad
    return g


# get normalized intensity histograms for each cell 
def intensity_hist(mask, img, bg, file_id, label, n_bins=10, percent_thresh=0.5, 
                   bg_correct=True, 
                   convexhull=False, plotfig=False):
    
    labels, n_labels = measure.label(mask, background = 0,return_num = True)
    cells = measure.regionprops(labels)  
    
    data = []
    cluster_hists = []
    cell_id = []
    #print("number of regions to start: {}".format(len(cells)))
    for kk, cell in enumerate(cells):
        
        # only look at regions that resemble the width of a cell to 
        # avoid any missegmented region
        cell_width = cell.minor_axis_length
        # if cell_width <= 15 and cell_width > 8 
        if cell_width <= 13 and cell_width >= 7: 
            # assign cell ID to be the same label of the mask
            cell_label = mask[int(cell.centroid[0]), int(cell.centroid[1])]
           # print("cell label",cell_label)
            # get mask region, but exclude surrounding cells in the roi
            ex = 1 
          
            cell_mask_og = mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
          
            cell_mask_og[cell_mask_og != cell_label] = 0   
            cell_mask_og[cell_mask_og == cell_label] = 1
            
            
        
            # only look at cells that have a non zero label
            #### try to look into why some cells get a zero label!!!!!
            if cell_label != 0:# and len(cell_mask_og) > 0:
                
                # store the cell label
                cell_id.append(cell_label)
                
                # convext hull the cell to fix small misegmentations
                if convexhull == True:
                    mask_hull = morphology.convex_hull_image(cell_mask_og)
                elif convexhull == False:
                    mask_hull = cell_mask_og
               
                # select the cell region for the fluorescence channel
                # combine the convex hull mask and the fluorescence channel roi
                
                cell_region = img[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
                cellfluor = cell_region * cell_mask_og#mask_hull
                
                # gather all of the pixel intensities within the cell
                cellpixvals = []
                cellfluor_flat = cellfluor.flatten()
                for pix in cellfluor_flat:
                    if pix > 0:
                        if bg_correct == True:
                            cellpixvals.append(pix - bg)
                        elif bg_correct == False:
                            cellpixvals.append(pix)
                            
                            
                # normalize the pixel intensities and bin 
                cellpixvals_norm = [((pix-min(cellpixvals))/
                              (max(cellpixvals)-min(cellpixvals))) for pix in cellpixvals]
                
                # get GINI coefficient of distribution of pixel intensities
                # in the cell
                gini = gini_coeff(cellpixvals)
                gini_norm = gini_coeff(cellpixvals_norm)
                
                
                x, bins = np.histogram(cellpixvals_norm, bins=n_bins)
                global bincenters
                bincenters = 0.5 * (bins[1:] + bins[:-1])
                # print(x, len(cellpixvals_norm))
                y = x / len(cellpixvals_norm)
                
                
                # calculate the percent clustering
                cluster_thresh = [0.1, 0.3, 0.5, 0.7, 0.9]
                cluster_param = []
                for thresh in cluster_thresh:
                    
                    cluster_param.append((sum(y[:int(n_bins*thresh)])))
                 
                    
                 
                # classify cell as if with a focus or not
                cell.image[cell.image > 0] = 1
               
                cell_fluor = img[cell.bbox[0]-ex:cell.bbox[2]+ex,
                                 cell.bbox[1]-ex:cell.bbox[3]+ex] 
                
                nblobs, blob_mask, cyto_mask = findblobs(cell_fluor, 
                                   cell_mask_og, norm_thresh = 0.75,
                                   eccent_thresh = 0.90, plot=False)
                if nblobs > 0:
                    partition_coeff_avg, parition_coeff_tot = calc_partition_coeff(cell_fluor, blob_mask, cyto_mask, bg)
                else:
                    partition_coeff_avg, parition_coeff_tot = False, False
                    
                cluster_hists.append((np.asarray(y),nblobs))
                
                
                data.append((file_id, cell_label, cluster_param[0], 
                             cluster_param[1], cluster_param[2],
                             cluster_param[3], cluster_param[4], gini, gini_norm,
                             label, nblobs, partition_coeff_avg, parition_coeff_tot))
        
                # plot cell by cell 
                if plotfig == True:
                    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(7,3))
                    ax = axes.ravel() 
                    ax[0].set_title('cell #: {}'.format(cell_label))
                    ax[0].imshow(cellfluor, cmap='inferno')
                    ax[0].contour(mask_hull, levels=0, colors='w', 
                                  linestyle='dashed')
                    ax[0].get_xaxis().set_visible(False)
                    ax[0].get_yaxis().set_visible(False)
                
                    ax[1].plot(bincenters, y, 'o-', c='k')
                    ax[1].set_xlabel('Normalized intensity', fontsize=12)
                    ax[1].set_ylabel('Fraction of pixels', fontsize=12)
                    ax[1].set_ylim(0,1)

                    fig.tight_layout()
                    plt.show()

    return data, cluster_hists    


def calc_partition_coeff(img, blob_mask, cyto_mask, bg):
    
    # measure the blob(s) average intensity
    blob_labels, n_blobs = measure.label(blob_mask, background=0, return_num=True)
    blobs = measure.regionprops(blob_labels)
    blob_tot_pix = 0
    blob_tot_int = 0
    for blob in blobs:
        for coord in blob.coords:
            blob_tot_int += (img[coord[0]][coord[1]] - bg)
        blob_tot_pix += blob.area
    blob_avg_int = blob_tot_int / blob_tot_pix
    
    
    cyto_labels, n_cyto = measure.label(cyto_mask, background=0, return_num=True)
    cyto = measure.regionprops(cyto_labels)
    cyto_tot_pix = 0
    cyto_tot_int = 0
    for cyto_reg in cyto:
        for coord in cyto_reg.coords:
            cyto_tot_int += (img[coord[0]][coord[1]] - bg)
        cyto_tot_pix += cyto_reg.area
    cyto_avg_int = cyto_tot_int / cyto_tot_pix
    
    partition_coefficient_pix = blob_avg_int / cyto_avg_int
    partition_coefficient_tot = blob_tot_int / cyto_tot_int
  #  print("pix", blob_avg_int, cyto_avg_int, partition_coefficient_pix)
  #  print("tot", blob_tot_int, cyto_tot_int, partition_coefficient_tot)
    
    return partition_coefficient_pix, partition_coefficient_tot 
    
            
        
        
        
    


def findblobs(cell,cellmask, norm_thresh = 0.65, area_thresh = 4, 
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
        DESCRIPTION. The default is 4.
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



def plot_hists(cluster_hists):
    
    cluster_hists_arr = [[],[]]
    
    for dataset in cluster_hists:
        
        for arr in dataset:
            if arr[1] == 0:
                cluster_hists_arr[0].append(arr[0])
            elif arr[1] > 0:
                cluster_hists_arr[1].append(arr[0])
    
    fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(6.25,2.5), sharey=True,
                             dpi=300)
    ax = axes.ravel() 
    print(len(cluster_hists_arr[0]), len(cluster_hists_arr[1]))
    for ii, subset in enumerate(cluster_hists_arr):
        for line in subset:
        
            ax[ii].plot(bincenters, line, '-', c='black', alpha=0.05, lw=0.5)
           # ax[ii].plot(bincenters, line, '-', c='gray', alpha=0.2, lw=1)
    
        
    for jj, set_ in enumerate(cluster_hists_arr):
        avg_clust_hist = []
        std_clust_hist = []
    
        clust_hist_arr = np.array([np.array(xi) for xi in set_])
   # print(np.shape(clust_hist_arr))
        for nn,el in enumerate(set_[0]):
            avg_clust_hist.append(np.mean(clust_hist_arr[:,nn]))
            std_clust_hist.append(np.std(clust_hist_arr[:,nn]))
        
   
        avg_clust_hist = np.asarray(avg_clust_hist)
        std_clust_hist = np.asarray(std_clust_hist)
        
        ax[jj].plot(bincenters, avg_clust_hist, '--', c='maroon', alpha=1, linewidth=1.5)
        ax[jj].fill_between(bincenters, avg_clust_hist - std_clust_hist,
                       avg_clust_hist + std_clust_hist, alpha=0.5, color='maroon')

        ax[jj].set_xlabel('Normalized intensity', fontsize=font_size)
        
        ax[jj].set_ylim(0,1)
 #      ax[0].legend(['Clustered (n={})'.format(clustered_count)])    
        #ax[1].set_xlabel('Normalized intensity', fontsize=font_size)
        ax[jj].set_xlim(0,1)
        #ax[0].legend(['n={}'.format(len(cluster_hists_arr))], fontsize=font_size)    
        ax[jj].tick_params(axis='y',labelsize=font_size)
        ax[jj].tick_params(axis='x',labelsize=font_size)
    ax[0].set_ylabel('Fraction of pixels', fontsize=font_size)
    fig.tight_layout()
        #plt.savefig(directory[:-5] + '20220816_Figure_condensation.png', dpi=300)
    plt.show() 


    
    


bg_correct = True
label = 'cI_agg'
font_size = 16



directory = sys.argv[1]
files = filepull(directory)
phasemask_files = [file for file in files if "_masks.tif" in file]
fluor_files = [file for file in files if "_masks" not in file]
phasemask_files.sort()
fluor_files.sort()



data = [[] for i in range(13)]
hists = []

for ii, file in enumerate(fluor_files[:]):
    print(file,phasemask_files[ii])
    fl_img = read_TIFFStack(file)[1]
    mask = read_TIFFStack(phasemask_files[ii])
    
    if bg_correct == True:
        bg_int = find_background_int(mask, fl_img) 
    else:
        bg_int = 0
    new_img = fl_img.copy()
       
    output, clustered_hists = intensity_hist(mask, new_img, bg_int, ii+1, label)
    
    hists.append(np.array(clustered_hists))
    
    for out in output:
        for jj, res in enumerate(out):
            data[jj].append(res)

plot_hists(hists)
name = 'PopTag_LL'
df = pd.DataFrame(data).transpose()
df.columns = ['FILE_ID', 'CELL_ID', 'cluster 0.1',
              'cluster 0.3', 'cluster 0.5',
              'cluster 0.7','cluster 0.9','GINI coefficient', 'GINI norm', 
              'Label', "Focus", "PC_avg", "PC_tot"]
print(df)
df.to_csv(directory[:-5] + label + '_clusteranalysis.csv', index = False) 
