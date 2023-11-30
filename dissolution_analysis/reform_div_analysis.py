# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:29:10 2023

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
from tqdm import tqdm
import matplotlib.patches as patches
from cellpose import models, io, plot
#from cellpose_omni import MODEL_NAMES
from cellpose_omni import models, core, plot
import seaborn as sns
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
    
    blur_img = filters.gaussian(img, sigma=.1)# blur image before cellpose
  
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
        

def plot_mask(img, mask, pre_process):
    
    fig, ax = plt.subplots(ncols=3, figsize=(6, 3), 
                           sharey=True, sharex=True)
    ax[0].imshow(frame, cmap='gray')
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


def findblobs(cell,cellmask, norm_thresh = 0.85, area_thresh = 6, 
              eccent_thresh = 0.8, plot=True):
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
    cell_blur = cellmask*filters.gaussian(cell, 1)
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
 
    if blob_count > 0:
        blob_bool = True
    else:
        blob_bool = False
    
    
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
    cell_keep_ans = input("Keep cell? (y/n) ")   
    if cell_keep_ans == 'y':
        cell_keep = True
    else:
        cell_keep = False
    
    return blob_bool, cell_keep


def verify_focus(image, focus):
    
    blob_mask = np.zeros((image.shape[0], image.shape[1]))
    y, x, r = focus
    
    blob_grid = create_circle_mask(image.shape[0], image.shape[1], center = (x,y), radius=1.5*r)
    blob_mask[blob_grid] = 1
    masked_fluor = blob_mask * cell_fluor
    out_arr = masked_fluor[np.nonzero(masked_fluor)]
    
    # mean of focus intensity
    mean_focus_intensity = np.mean(out_arr)
    
    keep = True
    
    return keep
    
    


def cell_total_intensity(coords, image, image_crop):
    
    
    pixel_intensities = []
   
    for coord in coords:
        pixel_intensities.append(image[coord[0], coord[1]])# - np.median(image_crop))
        
    return sum(pixel_intensities)
        
    

def create_circle_mask(h,w, center, radius):
    
    Y, X =np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


# foldername that contains all of the fits.mat files
directory = sys.argv[1]#'/Volumes/jsbiteen/Lab_Members/Chris_Azaldegui/Project_Hn/SPT/2023/2023-04-12/*.tif'
# pull in all of the fits.mat files
files = filepull(directory)

tif_files = [file for file in files if '.tif' in file]
#print(tif_files)
print("{} .tif files found".format(len(files)))
tif_files.sort()
#---------------------------------------


model_name = 'bact_phase_omni'
model = models.CellposeModel(gpu=False, model_type=model_name)


#-----------------------------------------

data = []
data_norm = []

daughter_blob = []

font_size = 13

f = 1
for ii,file in enumerate(tif_files[:]):
    
    
    stack = tif.imread(file)
    print(file)
    
    phase_stack = stack[:,1,:,:]
    fluor_stack = stack[:,0,:,:]
    
    
    # mask phase mask for stack 
    # invert phase stack 
    inverted_phase_stack = invert_img_stack(phase_stack)
    inverted_phase_stack = invert_img_stack(inverted_phase_stack)
    mask_stack = []
    for frame in inverted_phase_stack:
        
        segmented_cells, img_process = segment_cells(model, frame,
                                                     user_diam = None) 
        plot_mask(frame, segmented_cells, img_process)
        mask_stack.append(segmented_cells)
        
        
        
    blob_bool = True
    cell_data = []
    avg_fluor = []
    
    mother_cell_average = 0
    for jj, image in enumerate(fluor_stack):
        
        labels, n_labels = measure.label(mask_stack[jj], background = 0, return_num = True)
        cells = measure.regionprops(labels)
        
        ex = 2
        for kk, cell in enumerate(cells[:]):
            
            t = jj # in frame index
            
            temp_mask = mask_stack[jj].copy() # use temporary mask to prevent issues
    
            cell_origin = cell.centroid # define the centroid of the cell
            cell_label = mask_stack[jj][int(cell_origin[0]), int(cell_origin[1])] # deterine the label of cell 
            #print("cell label", cell_label)

            cell_mask = temp_mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the mask to the specific cell
            cell_mask[cell_mask != cell_label] = 0 # remove any other cells in the ROI
            cell_mask[cell_mask == cell_label] = 1 # change cell value to 1 
            
            # bg sub
            background = restoration.rolling_ball(image, radius = 100)
            image_bg = image - background
    
            cell_fluor = image_bg[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex] # crop the cb image as before
            
            
            
            blob_bool, cell_bool = findblobs(cell_fluor, cell_mask)
            
            
            if cell_bool == True:
                cell_area = cell.area
            
                total_cell_fluor = cell_total_intensity(cell.coords, image_bg, cell_fluor)
                avg_cell_fluor = total_cell_fluor / cell_area
                avg_fluor.append(avg_cell_fluor)
                
                if jj == 0:
                    blob_bool = False
                    mother_cell_avg = avg_cell_fluor
                    mother_cell_total = total_cell_fluor
                    daughter_blob.append([ii, "Mother", mother_cell_avg / mother_cell_avg, total_cell_fluor / mother_cell_total,
                                          total_cell_fluor, cell_area])
                else:
                    if blob_bool == True:
                        daughter_blob.append([ii,"Inherited",avg_cell_fluor / mother_cell_avg, total_cell_fluor / mother_cell_total,
                                              total_cell_fluor, cell_area])
                    elif blob_bool == False:
                        daughter_blob.append([ii,"Absent",avg_cell_fluor / mother_cell_avg, total_cell_fluor / mother_cell_total,
                                              total_cell_fluor, cell_area])
                        
                cell_data.append((cell_label, t, cell_area, total_cell_fluor, avg_cell_fluor, blob_bool))
    
            
    if len(avg_fluor) > 1:
        data_norm.append([f / avg_fluor[0] for f in avg_fluor])

        data.append(avg_fluor)
    
        cell_data_df = pd.DataFrame(cell_data)
        cell_data_df.columns = ['cell_label', 'time', 'cell_area', 'total_cell_fluor','avg_cell_fluor', 'blob_bool']
        print(cell_data_df)
  

x = ['File','Sample', 'Normalized Average Intensity', 'Normalized Cellular Intensity', 'Cellular Intensity', 'Cell Area']
all_data_df = pd.DataFrame(daughter_blob)
all_data_df.columns = x
print(all_data_df)
all_data_df.to_csv(directory[:-5] + 'reformation_by_division_results.csv', index=False)


fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(9, 5), 
                         sharey=True, sharex=True, dpi=100)

sns.set(style="whitegrid")


ax = sns.boxplot(ax = axes[0], x="Sample", y="Normalized Average Intensity", data=all_data_df, showfliers = False)
ax = sns.swarmplot(ax = axes[0], x="Sample", y="Normalized Average Intensity", data=all_data_df, color=".25")

ax = sns.boxplot(ax = axes[1], x="Sample", y="Normalized Cellular Intensity", data=all_data_df, showfliers = False)
ax = sns.swarmplot(ax = axes[1], x="Sample", y="Normalized Cellular Intensity", data=all_data_df, color=".25")


plt.show()

