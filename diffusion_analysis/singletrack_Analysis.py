# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:33:44 2021

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats as ss
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from math import log

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# read csv
def csv2df(file):
    '''
    Reads a .csv file as a pandas dataframe
    
    input
    -----
    file: string, csv filename
    
    output
    -----
    df: pd.dataframe
    '''
    df = pd.read_csv(file, header = 0)
    return df

# organize track data
# Determine total number of tracks
def total_tracks(df):
    n_tracks = df['TRACK_N'].max()
   
    return int(n_tracks)

# Separate tracks by ID into dictionary
def tracks_df2dict(df, n_tracks, min_length):
    
    tracks_dict = {} # initialize empty dictionary
    
    for ii in range(n_tracks): # for each track
        
        track = df.loc[df.TRACK_N == (ii)] # assign current track to var
        track_length = len(track) # find track length (n localizations)
   
        # if track length > or = min_length then store in dict
        if track_length >= min_length: 
            tracks_dict["track_{0}".format(ii)] = df.loc[df.TRACK_N == (ii)]
                  
    tracks_kept = len(tracks_dict) # how many track have min_length?
    #print("Total tracks to analyze: {}".format(tracks_kept)) # state it

    return tracks_dict


# calculate size of step
def calc_stepsize(x0,xF,y0,yF):
    x_step = xF - x0
    y_step = yF - y0
    stepsize_pix = np.sqrt(x_step**2 + y_step**2)
    stepsize_um = stepsize_pix*pixsize 
    return stepsize_um

# determine position in dictionary
def loc_in_dict(data,key,row,col,time_lag):
    loc0 = data[key].iloc[row,col]
    locF = data[key].iloc[row+time_lag,col]    
    return loc0,locF

def brownian_a(tau,D,a): # normal Brownian motion
    return log(4*D,10) + a*tau

def brownian(tau,D, sig): # normal Brownian motion with blurring
    return (8/3)*D*(tau) + 2*sig**2

# calculate MSD curves and apparent diffusion coefficient
def calc_MSD(track, tracks_dict, req_locs, plot=False):
    # corresponding column number for x and y coordinates    
    x_coord = 2
    y_coord = 1
    # determine # of localizations of track
    track_n_locs = len(tracks_dict[track])
    # determine # of steps of track
    track_n_steps = track_n_locs - 1 #req_steps
    # hold list for MSD of a track
    MSD_track = []        
    # start time_lag
    time_lag = 0
    # run a lag calc until max step num
    for lag in range(track_n_steps - 1): # changed the -1
        # keep count of time lag
        time_lag += 1
        # num of steps to calc
        n_steps = track_n_steps - time_lag
        #global displacement_list
        sq_displacement_list =[]
        # for each step of time lag
        for j, step in enumerate(range(n_steps)):
            x0, xF = loc_in_dict(tracks_dict, track, j, x_coord,time_lag)
            y0, yF = loc_in_dict(tracks_dict, track, j, y_coord,time_lag)
            sq_displacement = (calc_stepsize(x0,xF,y0,yF))**2
            sq_displacement_list.append(sq_displacement)
        mean_sq_dis = np.average(sq_displacement_list)
        MSD_track.append(mean_sq_dis)
    
    # from MSD curves, estimate diffusion coefficient
    tau_list = [(x*framerate) + framerate for x in range(len(MSD_track))]  
    
    ydata = np.asarray(MSD_track[:5])
    xdata = np.asarray(tau_list[:5])
    # if len(MSD_track) >= 3:
    if fit_alpha == True:
      #  diff, _ = curve_fit(brownian_a, [log(y,10) for y in tau_list[:8]],
       #                                 [log(y,10) for y in MSD_track[:8]],
        #                                  maxfev = 10000)
        a, diff = np.polyfit([log(y,10) for y in tau_list[:5]],
                             [log(y,10) for y in MSD_track[:5]], 1)
        # return diffusion coefficient
        return diff, a
    elif fit_alpha == False:
        diff, pcov = curve_fit(brownian, xdata, ydata,
                            maxfev = 10000)
       
        corr_matrix = np.corrcoef(ydata, brownian(xdata,*diff))
        corr_yx = corr_matrix[0,1]
        r_squared = corr_yx**2
        # return diffusion coefficient
      
        return diff[0], diff[1], 1, r_squared, (xdata,ydata)

        # calculate track radius of gyration
def get_rad_gyration(track, tracks_dict, plot = False):
    # corresponding column number for x and y coordinates    
 #   print(tracks_dict[track])
    x_coord = 2
    y_coord = 1
    # store x and y coordinates (pix)
    xcoordinates = []
    ycoordinates = []
    # determine track length
    track_length = len(tracks_dict[track])
    # for each localization in track store coordinates to find avg. position
    for loc in range(track_length):
        # store x,y in lists
        xcoordinates.append(tracks_dict[track].iloc[loc, x_coord])
       # print(tracks_dict[track].iloc[loc, x_coord])
        ycoordinates.append(tracks_dict[track].iloc[loc, y_coord])  
    # calculate average position of particle in trajectory
    x0 = np.average(xcoordinates)
    y0 = np.average(ycoordinates)
    # store radii for an individual track
    track_radii = []
    # calculate distance of each track position to avg position
    proj_list = []
    for tt in range(track_length):
        # define track position at frame
        xf = tracks_dict[track].iloc[tt,x_coord]
        yf = tracks_dict[track].iloc[tt,y_coord]
        # calculate radii
        radii = ((((x0-xf)**2 + (y0-yf)**2))**(0.5))*pixsize*1000
        # store radii in list
        track_radii.append(radii)     
        proj = ((xf-x0)*pixsize, (yf-y0)*pixsize)
        proj_list.append(proj)
    # calculate average radii
    radius_gyr = np.average(track_radii)
    # return average radius of gyration
    return radius_gyr, (x0,y0), track_radii, proj_list

# Calculate displacements
def calc_displacements(tracks_dict, plot = False):
    # holds lists for each track containing displacements
    all_step_size = []
    tracks_step_size = []
    all_track_vectors = [[] for ii in range(len(tracks_dict))]
    # corresponding column number for x and y coordinates
    x_coord = 2
    y_coord = 1
    # iterate through each track
    for i, track in enumerate(tracks_dict):
        track_steps = []
        # determine # of localizations of track
        track_n_locs = len(tracks_dict[track])
        # determine # of steps of track
        track_n_steps = track_n_locs - 1
        # for each step in track calc displacement
        for j, step in enumerate(range(track_n_steps)):
            # time_lag: step size for displacement calculation
            step = 1
            # determine init and final positions
            x0,xF = loc_in_dict(tracks_dict, track, row = j, 
                                col = x_coord, time_lag = step)
            y0,yF = loc_in_dict(tracks_dict, track, row = j,
                                col = y_coord, time_lag = step)
            # calculate and store the step vector
            step_vector = (xF-x0, yF-y0)
            all_track_vectors[i].append(step_vector)
            # calc displacement in microns
            step_size = (calc_stepsize(x0,xF,y0,yF))
            all_step_size.append(step_size) # store in all list
            track_steps.append(step_size)
        # store step size to corresponding track
        tracks_step_size.append(track_steps)
    # for each track calculate its average step size
    tracks_stepsize_avg = []
    for track in tracks_step_size[:]:
            avg_stepsize = np.average(track)
            tracks_stepsize_avg.append(avg_stepsize)
    # return cumulate step sizes, track avg step sizes, and track step vectors
    return all_step_size,tracks_stepsize_avg, all_track_vectors  


def calc_anisotropy(track, plot=False):
    import math
    # convert track step vector list to dataframe
    data = pd.DataFrame(np.asarray(track))
   
    # assign dataset to variable x
    x = data.loc[:,:].values
    # rescale the datasets to a standard normal distribution
    #normdata = StandardScaler().fit_transform(data)
   
    # run PCA
    pca = decomposition.PCA(n_components = 2)
    principalComponents = pca.fit_transform(data)  
    principalDF = pd.DataFrame(data = principalComponents,
                               columns = ['PC1', 'PC2'])
    
    Lmax = pca.explained_variance_[0]
    Lmin = pca.explained_variance_[1]
    
    # if variances are NaN change value to something very small to avoid problems
    if math.isnan(Lmax)==True:
        Lmax = 10**-3
    if math.isnan(Lmin)==True:
        Lmin = 10**-3

    # calculate anisotropy parameter
    #print(x_var, y_var)
    anisotropy = (Lmax - Lmin) / (Lmax + Lmin)
   
    # turn variances into percentages
    pc_var = (round(Lmax*100,2),
              round(Lmin*100,2))
    # assign PC points to variables
    x = principalComponents[:,0]
    y = principalComponents[:,1]
    return anisotropy
        


# plot the data    
def plot_data(dcoeffs):
    
    # no. of bins for all datasets
    n_bins = 50
    # initiate figure
    
    fig = plt.figure()
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    
    plt.rcParams['figure.figsize'] = 3,3
  
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title("App. Diff. Coef. (n={})".format(len(dcoeffs)), 
                    fontsize=10, fontweight='bold')
    plt.xlabel('D_app (um^2/s)', fontsize=10, fontweight='bold')
    plt.ylabel('Probability', fontsize=10, fontweight='bold')
    data_hist, bins = np.histogram(dcoeffs, bins=n_bins)   
    logbins = np.logspace(np.log10(10**-5), np.log10(10), n_bins)
    x, bins, p = plt.hist(dcoeffs, bins=logbins, color='g', 
                            edgecolor='k', alpha=0.75)
    plt.xscale('log')
    plt.yscale('linear')
    for item in p:
        item.set_height(item.get_height()/sum(x))
    #plt.ylim(0,40)
    #plt.xlim(0, 60)
    fig.tight_layout()
    plt.show()
    
    
    fig.tight_layout()
    plt.show()
    

def main():
    
    # set global parameters:
    global framerate, pixsize, scope, set_name, min_locs, fit_alpha, max_tau
    
    framerate = 0.040 # frame acquisition every x seconds
    min_track_len = 7 # in localizations
    pixsize = 0.049 # microns per pixel
    max_tau = 5
    fit_alpha = False
    Label = 'McdB-PS-_4hr'
    Set_number = '3'

    TracksFolder = sys.argv[1] # tracks file name
    print(TracksFolder) # print file name to verify
    FolderName = TracksFolder[:TracksFolder.find('/')]# folder name for output
    
    TrackFiles = filepull(TracksFolder) # pull in track csv files
    
    # initiate holding lists
    TracksDFs = []
    MasterTracks = []
    MasterIDs = [] # file ID
    # run through each track file to filter tracks
    for ii, File in enumerate(TrackFiles[:]):
        # convert csv to df
        TracksDF = csv2df(File)
        # how many tracks are in df?
        n_Tracks = total_tracks(TracksDF)
        # add index 
        FileIndex = [ii for x in range(len(TracksDF.index))]
        TracksDF['FILE_INDEX'] = FileIndex
        # add df to list
        TracksDFs.append(TracksDF)
        # convert df to dictionary and add to separate list
        TracksDict = tracks_df2dict(TracksDF, n_Tracks, min_track_len)
        MasterTracks.append(TracksDict)
        # how many were filtered
        print("file {}, initial {}: final {} ({}%)".format(ii, 
                                                  TracksDF['TRACK_N'].max(),
                                                  len(TracksDict), 
                                                  (len(TracksDict)/
                                                  TracksDF['TRACK_N'].max())*100))
    all_vectors = []
    for dataset in MasterTracks:
        _,_,vectors = calc_displacements(dataset)
        all_vectors.append(vectors)
        
    D_list= [] # apparent diffusion coefficient
    locpres_list = []
    alpha_list = []
    Rg_list= [] # apparent diffusion coefficient
    anisotropy_list = []
    TrackLengths = []
    Labels = []
    Set_numbers = []
    blob_class = []
    all_radii = []
    all_proj = []

    n_track = 0 # start count of number of tracks analyzed
    fig, axes = plt.subplots(figsize=(5,2), ncols=2,nrows=1,
                               sharey=True, sharex=True, dpi=300)
    ax = axes.ravel()
    msd_avg_in = []
    msd_avg_io = []
    msd_avg_out = []
    
    in_count = 0
    io_count = 0
    out_count = 0
    
    for ii,dataset in enumerate(MasterTracks[:]):
    # for each in track in the dataset
        for jj,track in enumerate(dataset):
            
            # determine track diffusion coefficient
            
            D, locpres, a, r2, msd_curve = calc_MSD(track, dataset, min_track_len)
            
            # plot MSD
            if dataset[track].iloc[0, 6] == 'in':
                ax[0].plot(msd_curve[0], msd_curve[1], color='blue', alpha = 0.1, 
                     linewidth=.5, linestyle='-')
                msd_avg_in.append(msd_curve[1])
                in_count += 1
            
           # elif dataset[track].iloc[0, 6] == 'io':
            #    ax[1].plot(msd_curve[0], msd_curve[1], color='blue', alpha = 0.05, 
             #        linewidth=.25, linestyle='-')
             #   msd_avg_io.append(msd_curve[1])
              #  io_count += 1
                
            
            elif dataset[track].iloc[0, 6] == 'out':
                ax[1].plot(msd_curve[0], msd_curve[1], color='blue', alpha = 0.1, 
                     linewidth=.5, linestyle='-')
                msd_avg_out.append(msd_curve[1])
                out_count += 1
            
            rg, coord, list_of_radii, proj_coords = get_rad_gyration(track, dataset)
            anis_param = calc_anisotropy(all_vectors[ii][jj])
            track_len_locs = len(dataset[track])
            if D > 0  and r2 >= 0.7:
               
               
                blob_class.append(dataset[track].iloc[0, 6])
                TrackLengths.append(track_len_locs)
                Labels.append(Label)
                Set_numbers.append(Set_number)
                D_list.append(D)
                locpres_list.append(locpres)
                alpha_list.append(a)
                Rg_list.append(rg)
                anisotropy_list.append(anis_param)
                all_radii.append(list_of_radii)
                all_proj.append(proj_coords)
                # determine the ID of the track 
                # TRACK_N is the 3rd column in csv
                MasterIDs.append(dataset[track].iloc[0,3])
            n_track += 1
            print("done with track {}:{}".format(ii,str(track)))
    
    
    msd_avg_in = np.asarray(msd_avg_in)
    msd_avg_io = np.asarray(msd_avg_io)
    msd_avg_out = np.asarray(msd_avg_out)
    
    ax[0].plot(msd_curve[0], np.mean(msd_avg_in,axis=0), color='black', alpha = 1, 
                     linewidth=2, linestyle='-')
    ax[0].fill_between(msd_curve[0], np.mean(msd_avg_in,axis=0) - ss.sem(msd_avg_in,axis=0), 
                          np.mean(msd_avg_in,axis=0) + ss.sem(msd_avg_in,axis=0), 
                          color = 'black', alpha = 0.25)
  #  ax[1].plot(msd_curve[0], np.mean(msd_avg_io,axis=0), color='black', alpha = 1, 
   #                  linewidth=1, linestyle='-')  
   # ax[1].fill_between(msd_curve[0], np.mean(msd_avg_io,axis=0) - ss.sem(msd_avg_io,axis=0), 
                         # np.mean(msd_avg_io,axis=0) + ss.sem(msd_avg_io,axis=0), 
                         # color = 'black', alpha = 0.25)
    ax[1].plot(msd_curve[0], np.mean(msd_avg_out,axis=0), color='black', alpha = 1, 
                     linewidth=2, linestyle='-')    
    ax[1].fill_between(msd_curve[0], np.mean(msd_avg_out,axis=0) - ss.sem(msd_avg_out,axis=0), 
                          np.mean(msd_avg_out,axis=0) + ss.sem(msd_avg_out,axis=0), 
                          color = 'black', alpha = 0.25)
    for ii in range(2):
        ax[ii].set_xlabel('tau (s)',fontsize=10)
        #ax[ii].set_yscale('log')
        #ax[ii].set_xscale('log')
        ax[ii].set_xlim(0.03, 0.22)
        if ii == 0:
            ax[ii].set_ylabel('MSD (\u03bcm\u00b2/s)',fontsize=10)
        
    fig.tight_layout()
    plt.show()
    
    datatosave = [MasterIDs,D_list,locpres_list, alpha_list, Rg_list, anisotropy_list, 
                  TrackLengths, Labels, Set_numbers, blob_class]
    dataDF = pd.DataFrame(datatosave).transpose()
    dataDF.columns = ['ID','D',"Localization precision",'alpha', 'Radius of Gyration', 'Anisotropy',
                      'Track_Len', 'Label', 'Set #', 'blob_class']
    
    '''
    # save all the radii
    all_radii = [item for sublist in all_radii for item in sublist]
    radii_labels = [Label for x in range(len(all_radii))]
    radii_DF = pd.DataFrame([all_radii, radii_labels]).transpose()
    radii_DF.columns = ["radius", 'Label']
    
    radii_DF.to_csv(TracksFolder[:-17] + Label + '_' + Set_number + 
                      '_mintracklen-{}'.format(min_track_len) + 
                  '_RadiiList.csv', 
                  index = True)
    
    all_proj = [item for sublist in all_proj for item in sublist]
    proj_x = [item[0] for item in all_proj]
    proj_y = [item[1] for item in all_proj]
    proj_DF = pd.DataFrame([proj_x, proj_y, radii_labels]).transpose()
    proj_DF.columns = ["x", 'y', 'Label']
    proj_DF.to_csv(TracksFolder[:-17] + Label + '_' + Set_number + 
                      '_mintracklen-{}'.format(min_track_len) + 
                  '_RadiiCoords.csv', 
                  index = True)
    '''
    print(dataDF)
    if fit_alpha == False:
        dataDF.to_csv(TracksFolder[:-10] + Label + '_' + Set_number + 
                      '_mintracklen-{}'.format(min_track_len) + 
                  '_SingleTrackAnalysis.csv', 
                  index = True)
    if fit_alpha == True:
        dataDF.to_csv(TracksFolder[:-11] + Label + '_' + Set_number + 
                      '_mintracklen-{}'.format(min_track_len) +
                  '_alphafit_SingleTrackAnalysis.csv', 
                  index = True)
        
main()