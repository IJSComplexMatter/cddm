#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""
from conf import SIZE, NFRAMES,DELTA

from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop, add
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iacorr_multi, normalize_multi, log_merge
from cddm.sim import simple_brownian_video
import numpy as np



#: this creates a brownian motion multi-frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame,)
video1 = simple_brownian_video(range(NFRAMES), shape = (SIZE+32,SIZE+32), delta = DELTA/2)

video2 = simple_brownian_video(range(NFRAMES), shape = (SIZE+32,SIZE+32), delta = DELTA*2, background = 0)

video = add(video1, video2)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SIZE), (0,SIZE//2)))

#: create window for multiplication...
window = blackman((SIZE,SIZE//2))

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

#: now perform auto correlation calculation with default parameters using iterative algorithm
data, bg, var = iacorr_multi(fft, range(NFRAMES))

#: inspect the data
viewer = MultitauViewer(scale = True, shape = (SIZE,SIZE//2))
viewer.set_data(data, bg, var)
viewer.set_mask(k = 25, angle = 0, sector = 30)
viewer.plot()
viewer.show()

#perform normalization and merge data
fast, slow = normalize_multi(data, bg, var, scale = True)
x,y = log_merge(fast, slow)

#: save the normalized data to numpy files
import numpy as np
#np.save("auto_correlate_multi_t.npy",x)
#np.save("auto_correlate_multi_data.npy",y)


