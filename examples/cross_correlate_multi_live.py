"""
Demonstrates how to compute fft of videos and the compute cross-correlation
function with the out-of-memory version of the multitau algorithm and do
live view of the computation.
"""
from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge

import numpy as np
from examples.conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX, DATA_PATH

#: see video_simulator for details, loads sample video
import examples.dual_video_simulator as dual_video_simulator
import importlib
importlib.reload(dual_video_simulator) #recreates iterator

t1, t2 = dual_video_simulator.t1, dual_video_simulator.t2 

#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

#:perform the actual multiplication
video = multiply(dual_video_simulator.video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

if __name__ == "__main__":
    import os.path as p

    #we will show live calculation with the viewer
    viewer = MultitauViewer(scale = True)
    
    #initial mask parameters
    viewer.k = 15
    viewer.sector = 30
    
    #: now perform auto correlation calculation with default parameters and show live
    data, bg, var = iccorr_multi(fft, t1, t2, period = PERIOD, viewer = viewer)

    #: save the normalized data to numpy files
    for norm in (1,2,3,5,6,7,9,10,11,13,14,15):
        fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
        if norm in (5,6,7):
            np.save(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_{}.npy".format(norm)),fast)
            np.save(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_{}.npy".format(norm)),slow)
        x,y = log_merge(fast, slow)
        
        np.save(p.join(DATA_PATH,"cross_correlate_multi_norm_{}_data.npy".format(norm)),y)
    np.save(p.join(DATA_PATH,"cross_correlate_multi_t.npy"),x)
    
    viewer.show()
