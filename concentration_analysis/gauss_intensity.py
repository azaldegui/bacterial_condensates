# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:04:57 2021

@author: azaldegc
"""


import sys
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import nct
from scipy.stats import gamma


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


# function to calculate intensity counts per molecule per frame
# given the Gaussian fit parameters
def gaussian_intensity(A,mu):
    
    intensity = 2*math.pi*A*mu**2
    
    return intensity

# function to convert intensity counts to photons
def counts_to_photons(counts):
    
    conv_gain = 1.42 # electrons per count
    em_gain_600 = 0.14*600 # true EM gain conversion given input multiplier
    
    photons = (counts * conv_gain) / em_gain_600
    
    return photons

# function to fit to a normal distribution
def fitnorm(data):
    
    w = 3.49*np.std(data)*len(data)**(-1/3)
    #binsize = int(max(data)/w)
    mu, sigma = norm.fit(data)
    nbins = len(np.arange(0, max(data), w))
    bin_edges = np.histogram_bin_edges(data, nbins)
    y = norm.pdf(bin_edges, mu, sigma)
    ypeak = round(bin_edges[np.argmax(y)],2)
    
    return nbins, y, ypeak

# function to fit to a T distribution
def fitt(data):
    
    w = 3.49*np.std(data)*len(data)**(-1/3)
    df, nc, loc, scale = nct.fit(data)
    nbins = len(np.arange(0, max(data), w))
    bin_edges = np.histogram_bin_edges(data, nbins)
    x = np.linspace(min(data), max(data), 100000)
    y = nct.pdf(x, df, nc, loc, scale)
    ypeak = round(x[np.argmax(y)],2)
    
    return nbins, y,x, ypeak  

# function to fit to a gamma distribution
def fitgamma(data):
    
    w = 3.49*np.std(data)*len(data)**(-1/3)
    nbins = len(np.arange(0, max(data), w))
    bin_edges = np.histogram_bin_edges(data, nbins)
    x = np.linspace(min(data), max(data), 100000)
    param = gamma.fit(data)
    y_gamma = gamma.pdf(x, *param)
    
    return nbins, y_gamma, x, round(x[np.argmax(y_gamma)],2)
    
    
    
    
    
# user defined parameters
bootstrap_n = 10
font_size=13
savename = '20220830_Ec_mCherry-McdB_all'
plot_dists = True
fit_gamma = True
fit_t = False
savetocsv = True

directory = sys.argv[1]
fits_files = filepull(directory)

intensities = [] # hold gaussian intensitie outputs
offsets = [] # hold offsets 
molecule_photons = [] # hold no of photons for each molecule
for ii,file in enumerate(fits_files):
    
    df = pd.read_csv(file, header = 0)

    for index,row in df.iterrows():
        
        goodfit = row['goodfit']
        if goodfit == 1:
            
            mu = row['widthc'] # deviation of the fit
            B = row['offset'] # offsete on the fit
            A = row['amp'] # amplitude of the fit
            
            gaussint = gaussian_intensity(A,mu) # calculate intensity counts
            photons = counts_to_photons(gaussint) # convert counts to photons
            
            molecule_photons.append(photons)
            intensities.append(gaussint)
            offsets.append(B)
            
if savetocsv == True:
    
    # prep data to export in csv
    datatosave = [molecule_photons]
    dataDF = pd.DataFrame(datatosave).transpose()
    dataDF.columns = ['No. Photons']
    print(dataDF)
    dataDF.to_csv(directory[:-9] + savename + '_Photons.csv', index = True)
    
print("Done calculating integrated gaussian intensities")            

bs_gammafit_out = []
bs_tfitmax = []
for m in range(bootstrap_n):
    
    # sampling dataset
    bsdata =  np.random.choice(molecule_photons, replace=True, size=int(len(intensities)*1)) 
    if fit_gamma == True:
        bins, y_gamma,x_gamma, gamma_max = fitgamma(bsdata)
        bs_gammafit_out.append(gamma_max)
        print(gamma_max)
    if fit_t == True:
        bins, y_t,x_t, max_t = fitt(bsdata)
        bs_tfitmax.append(max_t)
        #print(max_t)

    if plot_dists == True and fit_gamma==True:
        
            fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(6,3), 
                               sharey=False, sharex=False, dpi= 300)
            ax = axes.ravel() 
            for ll in range(2):
                
                ax[ll].tick_params(axis='x',labelsize=font_size)
                ax[ll].tick_params(axis='y',labelsize=font_size)
            gauss_ints = bsdata
        
            # histogram of counts
          #  ax[0].set_title('SM Int. Gauss. Intensity')
            ax[0].set_ylabel('Count', fontsize=font_size)
            ax[0].set_xlabel('Photons detected', fontsize=font_size)
            n_gi, bins_gi, patches_gi = ax[0].hist(gauss_ints, bins, 
                                       density=False, edgecolor='k')
            
            # gamma distribution fitting
            ax[1].set_ylabel('PDF', fontsize=font_size)
            ax[1].set_xlabel('Photons/localization', fontsize=font_size)
            n_gi, bins_gi, patches_gi = ax[1].hist(gauss_ints, bins, 
                                       density=True, edgecolor='k', linewidth=0.5)
            l = ax[1].plot(x_gamma, y_gamma, 'r--', linewidth=2)
            ax[1].set_title('Gamma Fit: max={}'.format(gamma_max), fontsize=10)
            ax[1].set_xlim(0, 1050)
            '''
            # t distribution fitting
            ax[2].set_ylabel('PDF', fontsize=font_size)
            ax[2].set_xlabel('Photons detected', fontsize=font_size)
            n_gi, bins_gi, patches_gi = ax[2].hist(gauss_ints, bins, 
                                       density=True, edgecolor='k')
            l = ax[2].plot(x_t, y_t, 'r--', linewidth=2)
            ax[2].set_title('T Fit: max={}'.format(max_t), fontsize=10)
            
            # offset values 
            ax[3].set_title('Fit Offsets')
            ax[3].set_xlabel('Offset', fontsize=font_size)
            ax[3].set_ylabel('Count', fontsize=font_size)
            bins_os, y_os, max_os = fitnorm(offsets)
            n_gi, bins_gi, patches_gi = ax[3].hist(offsets, bins_os, 
                                       density=True, edgecolor='k')
            l = ax[3].plot(bins_gi, y_os, 'r--', linewidth=2)
            ax[3].set_title('Offsets Norm. Fit: max={}'.format(max_os), 
                            fontsize=10)
            
            '''
            fig.tight_layout()
            plt.show()
    print("Done with bootstrap:{num:03d}".format(num=m+1))


mean = sum(bs_gammafit_out) / len(bs_gammafit_out)
variance = sum([((x - mean) ** 2) for x in bs_gammafit_out]) / len(bs_gammafit_out)
res = variance ** 0.5
st_error = res / np.sqrt(bootstrap_n)


t_mean = sum(bs_tfitmax) / len(bs_tfitmax)
t_variance = sum([((x - mean) ** 2) for x in bs_tfitmax]) / len(bs_tfitmax)
t_res = t_variance ** 0.5
t_st_error = t_res / np.sqrt(bootstrap_n)
print('gamma fit boostrapped (n={}) mean, stdev, st_error: {}, {}, {}'.format(bootstrap_n, 
                                                            round(mean,2), 
                                                            round(res,2),
                                                            round(st_error,2)))
  
    
print('t fit boostrapped (n={}) mean, stdev, st_error: {}, {}, {}'.format(bootstrap_n, 
                                                            round(t_mean,2), 
                                                            round(t_res,2),
                                                            round(t_st_error,2)))
