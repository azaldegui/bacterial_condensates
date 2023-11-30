# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:00:33 2021

@author: azaldegc


calculate the integrated fluorescence intensities of cells using a fluorescence
image and a phase mask

"""

import seaborn as sns
import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics as ss
import math as math
from skimage import (filters, measure, segmentation, util)


def filepull(directory):
    '''
    Finds files in directory and returns list of file names
    Parameters
    ----------
    directory : string
        Directory where the files are located.

    Returns
    -------
    filenames : list
        List of filenames in the directory.

    '''
    filenames = [img for img in glob.glob(directory)]   
    return filenames


def read_TIFFStack(stack_file):
    '''
    Reads TIFF stacks, returns stack and number of frames in stack

    Parameters
    ----------
    stack_file : tif.tif image
        An image or series of images to be converted into a numpy array.

    Returns
    -------
    stack : array
        A numpy array of an image or image series.
    n_frames : int
        The number of frames in stack or the Z dimension of 3D np array

    '''
    
    stack = tif.imread(stack_file) # read TIFF stack into a np array
    n_frames = stack.shape[0] # number of frames in stack
   
    return stack, n_frames


# function to 
def find_background_int(mask, img, bg='median'):
    '''
    Determine the background intensity of an image
    
    Parameters
    ----------
    mask : array
        DESCRIPTION.
    img : array
        DESCRIPTION.
    bg : string, optional
        DESCRIPTION. The default is 'median'.

    Returns
    -------
    background_intensities : list
        DESCRIPTION.

    '''
    
    inv_mask = np.invert(mask) # invert mask 
    
    # determine region that is the background
    labels, n_labels = measure.label(inv_mask, 
                                     background = 0, return_num=True )
    regions = measure.regionprops(labels)
    # collect all of the coordinates of the pixels in the background
    background_coords = regions[0].coords
    
    background_intensities = []    
    ints = []
    intensity = 0
    for coord in background_coords:
        intensity += img[coord[0], coord[1]]
        ints.append(img[coord[0], coord[1]])
    if bg == 'mean':
        background_intensities.append(intensity / len(background_coords))
    elif bg == 'median':
        background_intensities.append(ss.median(ints))
            
    return background_intensities


def cell_intensity(mask, img, bg_int, file_id, condition):
    '''
    

    Parameters
    ----------
    mask : TYPE
        DESCRIPTION.
    img : TYPE
        DESCRIPTION.
    bg_int : TYPE
        DESCRIPTION.
    file_id : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.

    Returns
    -------
    cell_fluors : TYPE
        DESCRIPTION.

    '''
    

    labels, n_labels = measure.label(mask,
                                     background = 0,return_num = True)
    cells = measure.regionprops(labels, intensity_image=img) 
 
    cell_fluors = []
    
    for ii, cell in enumerate(cells):
        
        cell_label = mask[int(cell.centroid[0]), int(cell.centroid[1])]
        
        ex = 1 
        temp_mask = mask.copy()
        cell_mask_og = temp_mask[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
        
        cell_mask_og[cell_mask_og != cell_label] = 0   
        cell_mask_og[cell_mask_og == cell_label] = 1
        
        cell_region = img[cell.bbox[0]-ex:cell.bbox[2]+ex,cell.bbox[1]-ex:cell.bbox[3]+ex]
        cell_fluor = cell_region * cell_mask_og
        
        cell.image[cell.image > 0] = 1
       
        cell_fluor = img[cell.bbox[0]-ex:cell.bbox[2]+ex,
                         cell.bbox[1]-ex:cell.bbox[3]+ex] 
        
        mask_hull = cell_mask_og
        
        nblobs, blob_mask, cyto_mask = findblobs(cell_fluor, 
                           mask_hull, cell_label, norm_thresh = 0.75,
                           eccent_thresh = 0.99, plot=True)
        
        
        
        cell_integrated_intensity = 0
        
        
        if nblobs == 0:
            blob_bool = 0
        else:
            blob_bool = 1
        
        for coord in cell.coords:
            corr_pixint = img[coord[0],coord[1]] - bg_int[0]
            #print(img[coord[0],coord[1]], bg_int[0])
            if corr_pixint < 0:
                corr_pixint == 0
            cell_integrated_intensity += corr_pixint
        
        # determine the number of photons in the cell for this imaging frame
        photons_per_cell = (cell_integrated_intensity * conv_gain) / em_gain_10
        # determine the number of molecules in this imaging frame
        copy_num = photons_per_cell / (photons_molec_frame / 2)
        # calculate the volume of the cell
        cell_vol = cell_volume(cell.area, cell.perimeter, cell.major_axis_length,
                               cell.minor_axis_length)
        # calculate the concentration of molecules per cell (molecules per um cubed)
        copy_conc = copy_num / cell_vol 
        # convert the concentration to molarity
        molarity = (copy_num / avogrado) / (cell_vol / liters_per_cubic_um)
        micromolar = molarity * 10**(6)
       # print("uM ",micromolar)
        cell_fluors.append((file_id, cell_label, cell_integrated_intensity, 
                            photons_per_cell, copy_num, cell_vol, copy_conc, 
                            molarity, micromolar, 
                            blob_bool, condition, sample, experiment_no))   
    
    return cell_fluors



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
    cytoplasm = measure.label(np.invert(filter_one_mask)*cellmask, background = 0)
    cytoplasm_props = measure.regionprops(cytoplasm, intensity_image = cell)
    
    filter_two_mask = filter_one_mask.copy()
    # second filter : intensity of regions
    cell_blobs, n_cell_blobs = measure.label(filter_one_mask,  
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

def cell_volume(area, perimeter, length, width):
    
    # coefficients for quadratic formular where we are solving for r, the radius
    # of the spherical caps of the cell. Can also be approximated to be
    # ~ 0.5 * cell width. 
    a = math.pi
    b =  -1*perimeter
    c = area
    
    # solve quadratic
    sols = np.roots([a,b,c])
    sols.sort()
    # assign r to correct solution, confirmed since h+2r is ~ cell length
    r = sols[0] 
    h = (perimeter - 2*math.pi*r)/2
    r = r*pixsize
    h = h*pixsize
    
    # return volume in um^3
    return (4/3)*(math.pi*r**2) + (2*math.pi*r*h)


# calculated sing mol intensity
global photons_molec_frame, sample, experiment_no

sample = 'PopTag_LL'
experiment_no = 3
condition = '100 uM IPTG 2hr'

photons_molec_frame = 91.1 # for 40ms
pixsize = 0.049 # in microns
conv_gain = 1.42 # electrons per count
em_gain_10 = 0.14*10 # EM conversion
liters_per_cubic_um = 1*10**15
avogrado = 6.02*10**23
var_plot = 'uM'


# pull fluors
# pull masks
directory = sys.argv[1]
print(directory)
files = filepull(directory)
phasemask_files = [file for file in files if "_PhaseMask" in file]
fluor_files = [file for file in files if '_max' in file]
phasemask_files.sort()
fluor_files.sort()

# store intensities
data = [[] for i in range(13)]
# for each fluor
for ii, image in enumerate(fluor_files):
   # print(ii+1)
    # read fluor and mask
    fluor,_=read_TIFFStack(image)
    mask,_=read_TIFFStack(phasemask_files[ii])

    # calc fluor background
    bg_int = find_background_int(mask, fluor)
    bg_int = [0]
    # calc all cell intensities (correct for bg)
    print(ii)
    cell_intensities = cell_intensity(mask, fluor, bg_int, image, condition)
    for cell in cell_intensities:
        for jj,output in enumerate(cell):
            data[jj].append(output)
            

df = pd.DataFrame(data).transpose()
df.columns = ['FILE_ID', 'CELL_ID', 'CELL_INTENSITY','PHOTONS_PER_CELL',
              'COPY_NUM', 'CELL_VOL_um^3','Molecules_per_um^3','Molar','uM', 
              'BlobBool', 'Condition', 'Sample', 'Experiment_no']
print(directory)
df.to_csv(directory[:-5] + sample + str(experiment_no) + '_copynum.csv', index = False)  

print(df)
all_uM = df[var_plot]


'''

df = df.astype({var_plot: float, 'BlobBool': float})
Means = df.groupby('BlobBool')[var_plot].mean()
Std = df.groupby('BlobBool')[var_plot].std()
Ns = [len(x) for x in df.groupby('BlobBool')[var_plot]]
fig, axes = plt.subplots(ncols=1,nrows=2, figsize=(8,8), 
                               sharey=False, sharex=False, dpi=150)
ax = axes.ravel()
print("full population", df["uM"].describe())

ax[0].set_xlabel('mCherry-PopTag_LL (\u03bcM)')
ax[0].set_ylabel('Counts')
sns.histplot(ax=ax[0], data=df, x="uM", stat='count', bins=25)


sns.violinplot('BlobBool',var_plot, data=df, ax = ax[1], inner="points")
#plt.setp(axes.collections, alpha=.75)
ax[1].scatter(x=range(len(Means)),y=Means,c="r")
print("classfied", df.groupby('BlobBool')[var_plot].describe())

#axes.set_title('mCherry-McdB molecules per \u03bcm\u00b3')

ax[1].set_xlabel('No focus vs focus')
ax[1].set_ylabel('Apparent concentration (\u03bcM)')

plt.show()
'''
