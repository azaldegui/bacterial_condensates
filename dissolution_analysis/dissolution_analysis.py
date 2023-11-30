# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:15:22 2023

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

    

def detect_focus(image, mask, plot=False):
    
    
    image_blur = filters.gaussian(image, 1)
    
    
    
    cb_by_mask = image_blur * mask
    
    # detect carboxysomes using Laplacian of Gaussian function
    blobs_log = feature.blob_log(cb_by_mask, min_sigma = 1.5, max_sigma=3.5, threshold=.015)
    
    # blobs to keep
    blobs_kept = []
   
    for blob in blobs_log:
        
        keep_focus = verify_focus(image, blob)
        
        #if keep_focus == True:
        blobs_kept.append(blob)
   
    
    if plot == True:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
      
        ax[0].imshow(image, vmin=0, vmax=2500)
        
        ax[1].imshow(image_blur)
        
        for blob in blobs_kept:
            y, x, r = blob
            c = plt.Circle((x, y), 1.5*r, color='red', linewidth=1, fill=False)
            ax[1].add_patch(c)
        plot_titles = ["original", "blurred"]
        for hh, axx in enumerate(ax):
            axx.set_title(plot_titles[hh])
            axx.contour(cell_mask, zorder=1, levels =1, colors='white')
            
        plt.tight_layout()
        plt.show()
    
    if len(blobs_kept) > 0:
        blob_bool = True
    else:
        blob_bool = False
    
    return blob_bool



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
font_size = 13
sample = 'McdB'


store_data = []

f = 1
for ii,file in enumerate(tif_files[:5]):
    
    
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
        # plot_mask(frame, segmented_cells, img_process)
        mask_stack.append(segmented_cells)
        
        
        
    blob_bool = True
    cell_data = []
    avg_fluor = []
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
            
            blob_bool = detect_focus(cell_fluor, cell_mask)
            
            cell_area = cell.area
            
            total_cell_fluor = cell_total_intensity(cell.coords, image_bg, cell_fluor)
            avg_cell_fluor = total_cell_fluor / cell_area
            avg_fluor.append(total_cell_fluor)
            cell_data.append((sample, ii, cell_label, t, cell_area, total_cell_fluor, avg_cell_fluor, blob_bool))
            
    
            
    if len(avg_fluor) == 2:
        data_norm.append([f / avg_fluor[0] for f in avg_fluor])
        data.append(avg_fluor)
        
        
        store_data += cell_data
        cell_data_df = pd.DataFrame(cell_data)
        cell_data_df.columns = ['sample','file','cell_label', 'time', 'cell_area',
                                'total_cell_fluor','avg_cell_fluor', 'blob_bool']
        
        #print(cell_data_df)


df = pd.DataFrame(store_data)
df.columns = ['sample','file','cell_label', 'time', 'cell_area',
                        'total_cell_fluor','avg_cell_fluor', 'blob_bool']
print(df)
df.to_csv(directory[:-5] + sample + '_dissolution.csv', index = False)   

print("average norm difference", np.mean(data_norm, axis=0), np.std(data_norm, axis=0))
fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(12, 4), 
                         sharey=False, sharex=False) 
font_size = 15
ax = axes.ravel()
x = ['Pre', 'Post']
print("number of cells analyzed", len(data))
for vv, line in enumerate(data):
    ax[0].plot(x, line, linestyle='dashed', marker='o')
    ax[1].plot(x, data_norm[vv],linestyle = 'dashed', marker ='o')
ax[0].set_ylabel("Cellular Intensity", fontsize=font_size)
ax[0].set_xlabel("Focus Dissolution", fontsize=font_size)
#ax[0].ticklabel_format(axis='x', style='sci')

ax[1].set_ylabel("Normalized Cellular Intensity", fontsize=font_size)
ax[1].set_xlabel("Focus Dissolution", fontsize=font_size)

for xx in ax:
    
    xx.tick_params(axis='x',labelsize=font_size)
    xx.tick_params(axis='y',labelsize=font_size)


plt.show()
            

                
                
                
                
            
        
        
        
    
    
    
 