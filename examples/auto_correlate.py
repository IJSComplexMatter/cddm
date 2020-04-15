#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""

from cddm.viewer import DataViewer, CorrViewer
from cddm.video import multiply, normalize_video, crop, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.core import acorr, normalize, stats
from cddm.multitau import log_average
from cddm.multitau import iacorr_multi, normalize_multi, log_merge
from cddm.sim import simple_brownian_video, seed
import numpy as np

from conf import SIZE, NFRAMES,DELTA

#set seeds so that each run of the experiment is on same dataset
seed(0)

#: this creates a brownian motion multi-frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame,)
video = simple_brownian_video(range(NFRAMES), shape = (SIZE+32,SIZE+32), delta = DELTA, background = 200)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SIZE), (0,SIZE)))

#: create window for multiplication...
window = blackman((SIZE,SIZE))

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES

#:perform the actual multiplication
video = multiply(video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax =31, kjmax = 31)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

#load int numpy array
fft_array, = asarrays(fft, NFRAMES)

if __name__ == "__main__":

    #: now perform auto correlation calculation with default parameters 
    data = acorr(fft_array)
    bg, var = stats(fft_array)
    
    #: perform normalization and merge data
    data_lin = normalize(data, bg, var, scale = True)
    
    #: inspect the data
    viewer = DataViewer()
    viewer.set_data(data_lin)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
    
    #: change size, to define time resolution in log space
    x,y = log_average(data_lin, size = 16)
    
    #: save the normalized data to numpy files
    np.save("auto_correlate_t.npy",x)
    np.save("auto_correlate_data.npy",y)


