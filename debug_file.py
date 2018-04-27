#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:19:23 2018

@author: sebastian
"""

#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')    
except:
    print('Not IPYTHON')
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib qt')   
import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr
from caiman.components_evaluation import estimate_components_quality_auto
from caiman.motion_correction import motion_correct_oneP_rigid
from caiman.base.rois import register_ROIs

import os
import cv2
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')
import bokeh.plotting as bpl
bpl.output_notebook()

folder_1 = '/media/sebastian/MYLINUXLIVE/MT/CaImAn/mov_1.hdf5'
fname_1 = sorted([folder_1])#[folder+f for f in os.listdir(folder) if ('recording' in f and '.tif' in f)]

folder_2 = '/media/sebastian/MYLINUXLIVE/MT/CaImAn/mov_2.hdf5'
fname_2 = sorted([folder_2])

frate = 10 # movie frame rate
gSig = 5   # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = 10  # average diameter of a neuron
do_motion_correction = True

m_raw_1 = cm.load_movie_chain(fname_1)
m_raw_2 = cm.load_movie_chain(fname_2)
downsample_ratio = 0.2
offset_mov_1 = -np.min(m_raw_1[:100]).astype(np.float32)  # make the dataset mostly non-negative
offset_mov_2 = -np.min(m_raw_2[:100]).astype(np.float32)


m_orig_1 = m_raw_1.interactive_crop(
    gain=1, offset = -offset_mov_1, fr=30)   # play movie (press q to exit

print(m_orig_1.shape)

m_orig_2 = m_raw_2.interactive_crop(size = (m_orig_1.shape[1],m_orig_1.shape[2]),
   gain=1, offset = -offset_mov_2, fr=30)   # play movie (press q to exit

print(m_orig_2.shape)

offset_mov_1 = -np.min(m_orig_1[:100])
offset_mov_2 = -np.min(m_orig_2[:100])
m_orig_1.save('cropped_1.tif')
m_orig_2.save('cropped_2.tif')

print(m_orig_1.shape,m_orig_2.shape)

cm.concatenate([m_orig_1.resize(1, 1, downsample_ratio)-offset_mov_1,
                m_orig_2.resize(1, 1, downsample_ratio)-offset_mov_2], axis=2).play(fr=60, gain=1, magnification=1, offset=0)


try:
    dview.terminate() # stop it if it was running
except:
    pass

c, dview, n_processes = cm.cluster.setup_cluster(backend='local', # use this one
                                                 n_processes=16,  # number of process to use, if you go out of memory try to reduce this one
                                                 )

mc_1 = motion_correct_oneP_rigid('cropped_1.tif',                        # name of file to motion correct
                           gSig_filt = [gSig]*2,                 # size of filter, xhange this one if algorithm does not work 
                           max_shifts = [3,3],                   # maximum shifts allowed in each direction 
                           dview=dview, 
                           splits_rig = 10,                      # number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100) 
                           save_movie = True)                    # whether to save movie in memory mapped format

new_templ = mc_1.total_template_rig
plt.subplot(2,2,1)    
plt.title('Filtered template')
plt.imshow(new_templ)       #% plot template
plt.subplot(2,2,2)
plt.title('Estimated shifts')
plt.plot(mc_1.shifts_rig)     #% plot rigid shifts
plt.legend(['x shifts', 'y shifts'])
plt.xlabel('frames')
plt.ylabel('pixels')

mc_2 = motion_correct_oneP_rigid('cropped_2.tif',                        # name of file to motion correct
                           gSig_filt = [gSig]*2,                 # size of filter, xhange this one if algorithm does not work 
                           max_shifts = [3,3],                   # maximum shifts allowed in each direction 
                           dview=dview, 
                           splits_rig = 10,                      # number of chunks for parallelizing motion correction (remember that it should hold that length_movie/num_splits_to_process_rig>100) 
                           save_movie = True)                  # whether to save movie in memory mapped format

new_templ = mc_2.total_template_rig
plt.subplot(2,2,3)    
plt.title('Filtered template')
plt.imshow(new_templ)       #% plot template
plt.subplot(2,2,4)
plt.title('Estimated shifts')
plt.plot(mc_2.shifts_rig)     #% plot rigid shifts
plt.legend(['x shifts', 'y shifts'])
plt.xlabel('frames')
plt.ylabel('pixels')

plt.show()

bord_px_rig_1 = np.ceil(np.max(mc_1.shifts_rig)).astype(np.int)     #borders to eliminate from movie because of motion correction        
fname_new_1 = cm.save_memmap(mc_1.fname_tot_rig, base_name='memmap_', order = 'C') # transforming memoruy mapped file in C order (efficient to perform computing)

bord_px_rig_2 = np.ceil(np.max(mc_2.shifts_rig)).astype(np.int)     #borders to eliminate from movie because of motion correction        
fname_new_2 = cm.save_memmap(mc_2.fname_tot_rig, base_name='memmap_', order = 'C') # transforming memoruy mapped file in C order (efficient to perform computing)

# load memory mappable file
Yr_1, dims_1, T_1 = cm.load_memmap(fname_new_1)
Y_1 = Yr_1.T.reshape((T_1,) + dims_1, order='F')

Yr_2, dims_2, T_2 = cm.load_memmap(fname_new_2)
Y_2 = Yr_2.T.reshape((T_2,) + dims_2, order='F')

m_corr = cm.movie(Y_1)
downsample_ratio = 1.
offset_corr = -np.min(m_corr[:100])  # make the dataset mostly non-negative
m_corr.resize(1, 1, downsample_ratio).play(
gain=1, offset=offset_mov_1, fr=30, magnification=1)

plt.subplot(2,1,1)
# compute some summary images (correlation and peak to noise)
cn_filter_orig_1, pnr_orig_1 = cm.summary_images.correlation_pnr(m_orig_1,gSig=gSig, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
cn_filter_1, pnr_1 = cm.summary_images.correlation_pnr(Y_1, gSig=gSig, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter_orig_1,pnr_orig_1)
inspect_correlation_pnr(cn_filter_1,pnr_1)

plt.subplot(2,1,2)
# compute some summary images (correlation and peak to noise)
cn_filter_orig_2, pnr_orig_2 = cm.summary_images.correlation_pnr(m_orig_2,gSig=gSig, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
cn_filter_2, pnr_2 = cm.summary_images.correlation_pnr(Y_2, gSig=gSig, swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
inspect_correlation_pnr(cn_filter_orig_2,pnr_orig_2)
inspect_correlation_pnr(cn_filter_2,pnr_2)
plt.show()

#% compute metrics for the results (TAKES TIME!!)
final_size = np.subtract(mc_1.total_template_rig.shape, 2 * 3) # remove pixels in the boundaries
winsize = 100
swap_dim = False
resize_fact_flow = .2    # downsample for computing ROF

tmpl_rig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
    fname_1[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
    mc_1.fname_tot_rig[0], final_size[0], final_size[1],
    swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

# print crispness values
print('Crispness original: '+ str(int(crispness_orig)))
print('Crispness rigid: '+ str(int(crispness_rig)))

#%% plot the results of Residual Optical Flow
fls = [ mc_1.fname_tot_rig[0][:-5]+ '_metrics.npz']
        #mc.fname[0][:-5] + '_metrics.npz']

#print('Running time rigid motion: ' + str(t_rigid))
#print('Running time piecewise motion: ' + str(t_pw))

plt.figure(figsize = (20,10))
for cnt, fl, metr in zip(range(len(fls)),fls,['pw_rigid','rigid','raw']):
    with np.load(fl) as ld:
        print(ld.keys())
        print(fl)
        print(str(np.mean(ld['norms'])) + '+/-' + str(np.std(ld['norms'])) +
              ' ; ' + str(ld['smoothness']) + ' ; ' + str(ld['smoothness_corr']))
        
        plt.subplot(len(fls), 3, 1 + 3 * cnt)
        plt.ylabel(metr)
        try:
            mean_img = np.mean(
            cm.load(fl[:-12] + '.mmap'), 0)[12:-12, 12:-12]
        except:
            try:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.tif'), 0)[12:-12, 12:-12]
            except:
                mean_img = np.mean(
                    cm.load(fl[:-12] + '.hdf5'), 0)[12:-12, 12:-12]
                    
        lq, hq = np.nanpercentile(mean_img, [.5, 99.5])
        plt.imshow(mean_img, vmin=lq, vmax=hq)
        plt.title('Mean')
        plt.subplot(len(fls), 3, 3 * cnt + 2)
        plt.imshow(ld['img_corr'], vmin=0, vmax=.35)
        plt.title('Corr image')
        plt.subplot(len(fls), 3, 3 * cnt + 3)
        #plt.plot(ld['norms'])
        #plt.xlabel('frame')
        #plt.ylabel('norm opt flow')
        #plt.subplot(len(fls), 3, 3 * cnt + 3)
        flows = ld['flows']
        plt.imshow(np.mean(
        np.sqrt(flows[:, :, :, 0]**2 + flows[:, :, :, 1]**2), 0), vmin=0, vmax=0.3)
        plt.colorbar()
        plt.title('Mean optical flow')      


min_corr = .8 # min correlation of peak (from correlation image)
min_pnr = 10 # min peak to noise ratio
min_SNR = 3 # adaptive way to set threshold on the transient size
r_values_min = 0.85  # threshold on space consistency (if you lower more components will be accepted, potentially with worst quality)
decay_time = 0.4  #decay time of transients/indocator

cnm_1 = cnmf.CNMF(n_processes=n_processes, 
                method_init='corr_pnr',                 # use this for 1 photon
                k=70,                                   # neurons per patch
                gSig=(3, 3),                            # half size of neuron
                gSiz=(10, 10),                          # in general 3*gSig+1
                merge_thresh=.8,                        # threshold for merging
                p=1,                                    # order of autoregressive process to fit
                dview=dview,                            # if None it will run on a single thread
                tsub=2,                                 # downsampling factor in time for initialization, increase if you have memory problems             
                ssub=2,                                 # downsampling factor in space for initialization, increase if you have memory problems
                Ain=None,                               # if you want to initialize with some preselcted components you can pass them here as boolean vectors
                rf=(40, 40),                            # half size of the patch (final patch will be 100x100)
                stride=(20, 20),                        # overlap among patches (keep it at least large as 4 times the neuron size)
                only_init_patch=True,                   # just leave it as is
                gnb=16,                                 # number of background components
                nb_patch=16,                            # number of background components per patch
                method_deconvolution='oasis',           #could use 'cvxpy' alternatively
                low_rank_background=True,               #leave as is
                update_background_components=True,      # sometimes setting to False improve the results
                min_corr=min_corr,                      # min peak value from correlation image 
                min_pnr=min_pnr,                        # min peak to noise ration from PNR image
                normalize_init=False,                   # just leave as is
                center_psf=True,                        # leave as is for 1 photon
                del_duplicates=True)                    # whether to remove duplicates from initialization
cnm_1.fit(Y_1)

cnm_2 = cnmf.CNMF(n_processes=n_processes, 
                method_init='corr_pnr',                 # use this for 1 photon
                k=70,                                   # neurons per patch
                gSig=(3, 3),                            # half size of neuron
                gSiz=(10, 10),                          # in general 3*gSig+1
                merge_thresh=.8,                        # threshold for merging
                p=1,                                    # order of autoregressive process to fit
                dview=dview,                            # if None it will run on a single thread
                tsub=2,                                 # downsampling factor in time for initialization, increase if you have memory problems             
                ssub=2,                                 # downsampling factor in space for initialization, increase if you have memory problems
                Ain=None,                               # if you want to initialize with some preselcted components you can pass them here as boolean vectors
                rf=(40, 40),                            # half size of the patch (final patch will be 100x100)
                stride=(20, 20),                        # overlap among patches (keep it at least large as 4 times the neuron size)
                only_init_patch=True,                   # just leave it as is
                gnb=16,                                 # number of background components
                nb_patch=16,                            # number of background components per patch
                method_deconvolution='oasis',           #could use 'cvxpy' alternatively
                low_rank_background=True,               #leave as is
                update_background_components=True,      # sometimes setting to False improve the results
                min_corr=min_corr,                      # min peak value from correlation image 
                min_pnr=min_pnr,                        # min peak to noise ration from PNR image
                normalize_init=False,                   # just leave as is
                center_psf=True,                        # leave as is for 1 photon
                del_duplicates=True)                    # whether to remove duplicates from initialization

cnm_2.fit(Y_2)



plt.subplot(1,2,1)

crd_1 = cm.utils.visualization.plot_contours(cnm_1.A, cn_filter_1, thr=.1, vmax=0.95)

plt.subplot(1,2,2)
crd_2 = cm.utils.visualization.plot_contours(cnm_2.A, cn_filter_2, thr=.1, vmax=0.95)

matched_ROIs_1, matched_ROIs_2, non_matched_1, non_matched_2, performance = register_ROIs(cnm_1.A,cnm_2.A,dims_1)

plt.subplot(1,2,1)
crd3 = cm.utils.visualization.plot_contours(cnm_1.A[:,matched_ROIs_1], cn_filter_1, thr=.8, vmax=0.95)
plt.subplot(1,2,2)
crd4 = cm.utils.visualization.plot_contours(cnm_2.A[:,matched_ROIs_2], cn_filter_2, thr=.8, vmax=0.95)

matched_ROIs_1, matched_ROIs_2, non_matched_1, non_matched_2, performance = register_ROIs(cnm_1.A,cnm_2.A,dims_1,cn_filter_1,cn_filter_2)














