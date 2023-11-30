# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:33:27 2022

@author: azaldegc
"""

import matplotlib.pyplot as plt
import matplotlib as mlab
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import curve_fit

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# user defined parameters
fit = 'Mix' # gaussian mixture, do not change
n_comps = 1 # number of gaussian states to fit to 
bootstraps = 10 # number of boostraps to perform 
boots_percent = 1 # fraction of data to sample
n_bins = 10 # number of bins, for plotting
font_size = 15 # fontsize, for plotting
plot = True # for plotting results
common_norm_ = False
n_samples = 3


# data organization
directory  =sys.argv[1]
filenames = filepull(directory)
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
    samples.append(df['Label'].iloc[0])
all_data = pd.concat(dfs).reset_index()
#samples = list(set(samples))
#samples.reverse()
#samples.sort()
#samples = ['in', 'io', 'out']
print(samples)



mus = []
sigms = []
weighs = []

#colors = ['firebrick','goldenrod','orangered','dimgray']
                #  'goldenrod','coral','royalblue']
fig, axes = plt.subplots(ncols=1,nrows=n_samples, figsize=(3.5,8), 
                                 sharey=True, sharex=True, dpi=100)

fitcurves = []

ax = axes.ravel()
# for each sample label perform gaussian mixture with bootstrap
for ii, sample in enumerate(samples):
    
    # select data by label
    data = all_data[all_data['Label']==sample]
    # take the log of the Diffusion coefficients
    dapps = np.log10(data["D"].to_numpy().reshape(-1,1))
   # dapps = data["D"].to_numpy().reshape(-1,1)
    
    all_means = []
    all_weights = []
    for mm in range(bootstraps):
        # randomly sample data
        bs_dapp =  np.random.choice(dapps.ravel(), replace=True, 
                                    size=int(len(data)*boots_percent))
        # perform gaussian mixture fitting
        if fit == 'Mix':
            bs_dapp = bs_dapp.reshape(-1,1)
            gm = GaussianMixture(n_components=n_comps, 
                                 covariance_type='full').fit(bs_dapp)
            
            M_best = gm
            weights = gm.weights_.flatten()
            sigmas = [np.power(10,x) for x in gm.covariances_.flatten()]
            means = [np.power(10,x) for x in gm.means_.flatten()]
            
            if mm == bootstraps:
                mus.append(means)
                sigms.append(sigmas)
                weighs.append(weights)
                
    
        sort_means =  np.asarray(means).argsort()
        means.sort()
        weights = weights[sort_means]
        all_means.append(means)
        all_weights.append(weights)
        
    x = np.linspace(np.log10(5*10**-4), np.log10(10**1), 10000)#, min(dapps), max(dapps), 1000000)
    logprob = M_best.score_samples(x.reshape(-1,1))
    responsibilities = M_best.predict_proba(x.reshape(-1,1))   
    
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    fitcurves.append((dapps, x, pdf, pdf_individual))
    
    all_means = np.asarray(all_means).reshape(1,bootstraps,n_comps) 
    all_weights = np.asarray(all_weights).reshape(1,bootstraps,n_comps)
    
    mean_D = []
    stdev_D = []
    mean_pi = []
    stdev_pi = []
    # results
   
    for oo in range(n_comps):
        
        mean_D.append(np.mean(all_means[:,:,oo]))
        stdev_D.append(np.std(all_means[:,:,oo]))
        mean_pi.append(np.mean(all_weights[:,:,oo]))
        stdev_pi.append(np.std(all_weights[:,:,oo]))
        
    print(sample, "N = {}".format(len(dapps)))
    print(mean_D, stdev_D, mean_pi, stdev_pi)
    
    
    xxx, bins, p = ax[ii].hist(dapps, density = True, bins = n_bins,
                       histtype = 'bar', color = 'darkgoldenrod', alpha=0.88,
                       linewidth=1, edgecolor='white')

    
    
    line = ax[ii].plot(x, pdf, '-k', linewidth=3)
    indi_line = ax[ii].plot(x, pdf_individual, '--k', linewidth=3)
   
    ax[ii].set_ylim(0,1.4)
    ax[ii].set_xlim(np.log10(3*10**-4), np.log10(3*10**1))
    
    if ii == 2:
        ax[ii].set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   
    ax[ii].set_ylabel('Density', fontsize=font_size)
    ax[ii].tick_params(axis='x',labelsize=font_size)
    ax[ii].tick_params(axis='y',labelsize=font_size)

   

           
fig.tight_layout()
#plt.savefig(directory[:-5] + '20220926_Fig_Dapp_all.png', dpi=300) 
plt.show()

        
   
  

'''
fig, axes = plt.subplots(ncols=1,nrows=3, figsize=(6, 11), 
                                 sharey=True, sharex=False)             
ax = axes.ravel()                
for ll,samp in enumerate(fitcurves): 
    
    dapps = samp[0]
   
    x = samp[1]
    pdf = samp[2]
    pdf_individual = samp[3]
    
    
    xxx, bins, p = ax[ll].hist(dapps, density = True, bins = n_bins,
                       histtype = 'bar', color = colors[ll],
                       linewidth=1, edgecolor='white')

    
    
    #line = ax[ll].plot(x, pdf, '-k', linewidth=3)
    indi_line = ax[ll].plot(x, pdf_individual, '-k', linewidth=3)
   # factor = max(pdf) / max(fitcurves[0][2])
   # wt_dist = fitcurves[0][3] * factor
    if ll == 2 or ll == 1:
        wt_line = ax[ll].plot(fitcurves[0][1], fitcurves[0][3], linestyle='--',color = 'navy',
                          alpha=1, linewidth=3)
   
    ax[ll].set_ylim(0,1)
    #ax[ll].set_xlim(-4,1)
    
    ax[ll].set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   # if ll == 0 or ll == 2:
    ax[ll].set_ylabel('Density', fontsize=font_size)
    ax[ll].tick_params(axis='x',labelsize=font_size)
    ax[ll].tick_params(axis='y',labelsize=font_size)
fig.tight_layout()    
plt.savefig(directory[:-5] + '20220815_Fig_Dapp_all.png', dpi=300)     
plt.show()
        
'''
    
   