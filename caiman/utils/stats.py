# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:49:57 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
from multiprocessing import Pool

try:
    import numba
except:
    pass

import numpy as np

#%%


def mode_robust_fast(inputData, axis=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """

    if axis is not None:

        def fnc(x): return mode_robust_fast(x)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        data = inputData.ravel()
        # The data need to be sorted for this to work
        data = np.sort(data)
        # Find the mode
        dataMode = _hsm(data)

    return dataMode
#%%


def mode_robust(inputData, axis=None, dtype=None):
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3
    """
    if axis is not None:
        def fnc(x): return mode_robust(x, dtype=dtype)
        dataMode = np.apply_along_axis(fnc, axis, inputData)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(data):
            if data.size == 1:
                return data[0]
            elif data.size == 2:
                return data.mean()
            elif data.size == 3:
                i1 = data[1] - data[0]
                i2 = data[2] - data[1]
                if i1 < i2:
                    return data[:2].mean()
                elif i2 > i1:
                    return data[1:].mean()
                else:
                    return data[1]
            else:

                wMin = np.inf
                N = data.size // 2 + data.size % 2

                for i in range(0, N):
                    w = data[i + N - 1] - data[i]
                    if w < wMin:
                        wMin = w
                        j = i

                return _hsm(data[j:j + N])

        data = inputData.ravel()
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()
        if dtype is not None:
            data = data.astype(dtype)

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        dataMode = _hsm(data)

    return dataMode

#%%
#@numba.jit("void(f4[:])")

def get_ROI_distances(cnm):
    Arr = (cnm.A.toarray()).reshape(list(cnm.pnr.shape[::-1])+[cnm.A.shape[-1]])
    X,Y = np.meshgrid(np.arange(Arr.shape[1]),np.arange(Arr.shape[0])) 
    dists = np.zeros([Arr.shape[-1]]*2)
    locs = np.zeros((Arr.shape[-1],2))
    for a in range(Arr.shape[-1]):
        A = Arr[:,:,a]
        A/=A.sum()
        cog_x = (A*Y).sum()
        cog_y = (A*X).sum() 
        
        locs[a] = [cog_x,cog_y]
        
    for a1 in range(locs.shape[0]):
        for a2 in range(locs.shape[0]):
            dists[a1,a2]=(((locs[a1,:]-locs[a2,:])**2).sum())**0.5
            
    return(dists,locs)

def _hsm(data):
    if data.size == 1:
        return data[0]
    elif data.size == 2:
        return data.mean()
    elif data.size == 3:
        i1 = data[1] - data[0]
        i2 = data[2] - data[1]
        if i1 < i2:
            return data[:2].mean()
        elif i2 > i1:
            return data[1:].mean()
        else:
            return data[1]
    else:

        wMin = np.inf
        N = old_div(data.size, 2) + data.size % 2

        for i in range(0, N):
            w = data[i + N - 1] - data[i]
            if w < wMin:
                wMin = w
                j = i

        return _hsm(data[j:j + N])
# Functions to calculate normalized correlation:

def normalized_correlation(spike_train_1,spike_train_2):
    return(np.inner(spike_train_1,spike_train_2)/((np.linalg.norm(spike_train_1)+np.linalg.norm(spike_train_2))/2))

def norm_corr_mat(spikes):
    corrmat = np.zeros([spikes.shape[0]]*2)
    for i in range(spikes.shape[0]):
        for j in range(spikes.shape[0]):
            corrmat[i,j]=normalized_correlation(spikes[i,:],spikes[j,:])
    return(corrmat)
    

def calc_x_rest(frames,length):
    x_rest = np.zeros((length))
    f = 0
    mode = 1
    for frame in frames:
        x_rest[f:]=mode
        f+=frame
        mode=1-mode
    return x_rest*2-1     
    

def yuste_sample(spikes):
    spike_copy = np.empty(spikes.shape)
    for j in range(spikes.shape[0]):
        spike_copy[j,:]=np.roll(spikes[j,:],np.random.randint(spikes[j,:].shape[0]))
    return norm_corr_mat(spike_copy)
        

def yuste_bootstrap(spikes,n_samples=10000,n_threads = 40):
    
    p = Pool(n_threads)
    
    spikes_shifted=spikes.copy()
    
    rnd_corrs = p.map(yuste_sample,[spikes_shifted]*n_samples)
    return np.concatenate([rnd_corrs])
    


