# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:18:59 2022

@author: azaldegc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# simulate cluster in cell at constant fraction of intensity 
# and varying cluster size
def sim_grid_size(x,y,f):
    
    # make 2D array and fill with random values from 0 to 5. 
    grid = np.random.randint(0,10, (y,x))
    print(np.average(grid))
    # for each row in the 2D array replace the first f fraction of pixels
    # with random high values ranging from 95 to 100
    for ii,r in enumerate(grid):
        for jj,e in enumerate(r[:int(len(r)*f)]):   
            grid[ii,jj] = np.random.randint(90,100)

    return grid



def norm_grid(grid, n_bins,p, plotfig=True):
    
     pixels = grid.flatten()
     
     pixels_norm = (pixels - min(pixels)) / (max(pixels) - min(pixels))
  
     x, bins = np.histogram(pixels_norm, bins=n_bins)
     global bincenters
     bincenters = 0.5 * (bins[1:] + bins[:-1])
     # print(x, len(cellpixvals_norm))
     y = x / len(pixels_norm)
     
     
     if plotfig == True:
         fig, axes = plt.subplots(nrows = 1, ncols=2, figsize=(9,2), dpi=300)
         ax = axes.ravel() 
         ax[0].set_title('parameter: {}'.format(round(p,1)))
         ax[0].imshow(grid, cmap='inferno', vmin=0, vmax=100)
        
         ax[0].get_xaxis().set_visible(False)
         ax[0].get_yaxis().set_visible(False)
         
         ax[1].plot(bincenters, y, 'o-', c='k')
         ax[1].set_xlabel('Normalized intensity', fontsize=12)
         ax[1].set_ylabel('Fraction of pixels', fontsize=12)
         ax[1].set_ylim(0,1)
         
         fig.tight_layout()
         plt.show()

     return y


     
 
f_vals = np.arange(0,1.1,0.1)

for ii,f in enumerate(f_vals[:]):
    
    cell = sim_grid_size(100,30,f)    
    
    intensity_hist = norm_grid(cell, 10, f)

def sim_int_dist(x,y,p):
    
    # make 2D array and fill with random values from 0 to 10. 
    grid = np.random.randint(0,10,(y,x))
    # keep fraction of pixels to be dense phase constant
    f = 0.2
    # change the range of dense phase pixel intensities
    for ii,r in enumerate(grid):
        for jj,e in enumerate(r[:int(len(r)*f)]):   
            grid[ii,jj] = np.random.randint(100*p,100*p+10)

    return grid

p_vals = np.arange(0,1.1,0.1)
print(p_vals)
for ii,p in enumerate(p_vals[:]):
    
    cell = sim_int_dist(100,30,p)   
   
    
    intensity_hist = norm_grid(cell, 10, p)
