#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:19:23 2018

@author: sebastian
"""

import h5py
import numpy as np
import caiman as cm
import matplotlib.pyplot as plt

#mov = cm.load('/home/sebastian/Downloads/mov.tif')
mov = cm.load('/media/sebastian/MYLINUXLIVE/MT/CaImAn/mov.hdf5')
#
#f = h5py.File('mov.hdf5','w')
#f.create_dataset("mov", mov.shape, compression="gzip")
#
#f['mov'][:]=np.array(mov)
#
#plt.imshow(f['mov'][500,:,:])
#
#f.close()
