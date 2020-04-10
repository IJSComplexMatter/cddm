#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""

from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge
from cddm.sim import simple_brownian_video, create_random_times1
import numpy as np
import matplotlib.pyplot as plt

nframes = 1024
#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(nframes,n = 16)

#: this creates a brownian motion multi-frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame,)
video = simple_brownian_video(t1,t2, shape = (512+32,512+32))

#: crop video to selected region of interest 
video = crop(video, roi = ((0,512), slice(0,512)))

#: apply dust particles
dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
dust2 = plt.imread('dust2.png')[...,0]
dust = ((dust1,dust2),)*nframes
video = multiply(video, dust)

#: create window for multiplication...
window = blackman((512,512))

#: we must create a video of windows for multiplication
window_video = ((window,window),)*nframes

#:perform the actual multiplication
video = multiply(video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax =37, kjmax = 37)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

#we will show live calculation with the viewer
viewer = MultitauViewer(scale = True)

#initial mask parameters
viewer.k = 15
viewer.sector = 30

#: now perform auto correlation calculation with default parameters and show live
data, bg, var = iccorr_multi(fft, t1, t2, period = 32, viewer = viewer)

#perform normalization and merge data
fast, slow = normalize_multi(data, bg, var,  scale = True)
x,y = log_merge(fast, slow)

#: save the normalized data to numpy files
for norm in (0,1,2,3):
    fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
    x,y = log_merge(fast, slow)
    np.save("cross_correlate_multi_norm_{}_t.npy".format(norm),x)
    np.save("cross_correlate_multi_norm_{}_data.npy".format(norm),y)


