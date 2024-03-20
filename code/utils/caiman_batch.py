# running caiman in batch
import cv2
import glob
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import json
from datetime import datetime

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt
from caiman.base.movies import play_movie

# script

# PFC NEURONS - GREAT ONE
fnames = [r'/Users/js0403/miniscope/PFC-Neurons/122D/AAV2/3-Syn-GCaMP8f/2024_02_06/12_21_26/miniscopeDeviceName',
          r'/Users/js0403/miniscope/PFC-Neurons/122D/AAV2/3-Syn-GCaMP8f/2024_03_05/12_51_33/miniscopeDeviceName',
          r'/Users/js0403/miniscope/PFC-Neurons/122D/AAV2/3-Syn-GCaMP8f/2024_03_15/10_57_51/miniscopeDeviceName',
          ]

def run_caiman(movie_path, min_corr, min_pnr, gSig_val, rf, stride):

    # Enter the directory containing your miniscope data and let magic happen!
    root_dir = movie_path; del movie_path

    # dont change
    contents_dir = os.listdir(root_dir) # get directory contents
    metadata_dir = os.path.join(root_dir,[i for i in contents_dir if '.json' in i][0]) # metadata files
    metadata = json.load(open(metadata_dir)) # load metadata
    fr = metadata['frameRate'] # define framerate

    # discover .avi or .tif 
    movie_chain = [i for i in contents_dir if '.avi' in i] # identify .tif or .avi files
    path_chain = [os.path.join(root_dir,i) for i in movie_chain] # combines your movie chain with the root directory to discover movies

    # reduce to native .avi save-out used by miniscope. Because our processing creates tifs, we can assume .avi is the original file
    movie_orig = cm.load_movie_chain(path_chain,fr=fr)

    # use this to save data later
    save_name = os.path.join(root_dir,'movie_chain')

    # CHANGE ME IF MOT CORRECTION ISNT WORKING
    gSig_filt = (10,10)      # sigma for high pass spatial filter applied before motion correction, used in 1p data, 10,10 for 122D

    # dataset dependent parameters
    frate = fr                     # movie frame rate
    decay_time = 0.4                 # length of a typical transient in seconds

    # motion correction parameters
    motion_correct = True    # flag for performing motion correction
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (stride, stride)     # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (int(stride/2), int(stride/2))  # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'      # replicate values along the boundaries

    mc_dict = {
        'fnames': path_chain,
        'fr': frate,
        'decay_time': decay_time,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }
    parameters = params.CNMFParams(params_dict=mc_dict)

    # start cluster
    try:
        cluster.terminate()
    except:
        pass
    cluster, n_processes = auto_cluster()

    # motion correct
    mot_correct = MotionCorrect(path_chain, dview=cluster, **parameters.get_group('motion'))
    mot_correct.motion_correct(save_movie=True)

    # save results
    fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='motioncorr_', order='C',
                            border_to_0=bord_px)

    # plot results
    bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)

    # parameters for source extraction and deconvolution
    p = 1               # order of the autoregressive system
    K = None            # upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array([gSig_val, gSig_val])  # expected half-width of neurons in pixels 
    gSiz = 4*gSig + 1     # half-width of bounding box created around neurons during initialization
    merge_thr = .7      # merging threshold, max correlation allowed
    rf = rf              # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = stride # amount of overlap between the patches in pixels 
    tsub = 2            # downsampling factor in time for initialization, increase if you have memory problems
    ssub = 1            # downsampling factor in space for initialization, increase if you have memory problems
    gnb = 0             # number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None  # None leaves background of each patch intact (use True if gnb>0)
    nb_patch = 0        # number of background components (rank) per patch (0 for CNMFE)
    min_corr = min_corr      # min peak value from correlation image
    min_pnr = min_pnr       # min peak to noise ration from PNR image
    ssub_B = 2          # additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = 1.4  # radius of ring is gSiz*ring_size_factor

    # print this out for the user
    print('Half-width of neuron:', gSig[0])
    print('Half-width of bounding box around neurons during init.:', gSiz[0])

    parameters.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                    'K': K,
                                    'gSig': gSig,
                                    'gSiz': gSiz,
                                    'merge_thr': merge_thr,
                                    'p': p,
                                    'tsub': tsub,
                                    'ssub': ssub,
                                    'rf': rf,
                                    'stride': stride_cnmf,
                                    'only_init': True,    # set it to True to run CNMF-E
                                    'nb': gnb,
                                    'nb_patch': nb_patch,
                                    'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                    'low_rank_background': low_rank_background,
                                    'update_background_components': True,  # sometimes setting to False improve the results
                                    'min_corr': min_corr,
                                    'min_pnr': min_pnr,
                                    'normalize_init': False,               # just leave as is
                                    'center_psf': True,                    # True for 1p
                                    'ssub_B': ssub_B,
                                    'ring_size_factor': ring_size_factor,
                                    'del_duplicates': True,                # whether to remove duplicates from initialization
                                    'border_pix': bord_px});                # number of pixels to not consider in the borders)


    # load memory mappable file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = Yr.T.reshape((T,) + dims, order='F')

    # start cluster
    try:
        cluster.terminate()
    except:
        pass
    cluster, n_processes = auto_cluster()

    # fit model
    cnmfe_model = cnmf.CNMF(n_processes=n_processes, 
                            dview=cluster, 
                            params=parameters)
    cnmfe_model.fit(images)

    # get datetime
    id = dateID()

    # save data
    print("Saving CNMFE model...")
    save_path =  save_name+'_'+id+'_cnmfe.hdf5'
    cnmfe_model.estimates.Cn = corr # squirrel away correlation image with cnmf object
    cnmfe_model.save(save_path)

    print("Saved to:",save_path)

def auto_cluster():
    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
    num_processors_to_use = None

    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=num_processors_to_use, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")
    return cluster, n_processes

def dateID():
    id = ''.join(''.join(''.join(datetime.now().strftime("%Y-%m-%d %H:%M:%S").split(':')).split(' ')).split('-'))
    return id

def params():
    min_corr=0.8; min_pnr=4; gSig_val=6; rf=200; stride=int(rf/2)
    return min_corr, min_pnr, gSig_val, rf, stride

def run_batch(fnames):
    # run in batch
    for i in fnames:
    
        # prep for running caiman
        movie_path = i
        min_corr = 0.8
        min_pnr = 4
        gSig_val = 6
        rf = 200
        stride = int(rf/2)
    
        print("running caiman on",i)
        run_caiman(movie_path,min_corr,min_pnr, gSig_val, rf, stride)
        print("Finished running caiman on",i)

# get critical parameters
min_corr, min_pnr, gSig_val, rf, stride = params()
