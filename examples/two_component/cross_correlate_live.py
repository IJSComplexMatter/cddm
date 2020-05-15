"""
Demonstrates how to compute fft of videos and the compute cross-correlation
function with the out-of-memory version of the multitau algorithm and do
live view of the computation.
"""
from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop, load
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge

import numpy as np
from conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX

#: see video_simulator for details, loads sample video
import dual_video_simulator
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

    #we will show live calculation with the viewer
    viewer = MultitauViewer(scale = True,semilogx =  True)
    
    #initial mask parameters
    viewer.k = 15
    viewer.sector = 30
    
    #: now perform auto correlation calculation with default parameters and show live
    data, bg, var = iccorr_multi(fft, t1, t2, period = PERIOD, viewer = viewer)
    
    #perform normalization and merge data
    fast, slow = normalize_multi(data, bg, var,  scale = True)
    x,y = log_merge(fast, slow)
    
    #: save the normalized data to numpy files
    for norm in (0,1,2,3,4,5,6,7):
        fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
        x,y = log_merge(fast, slow)
        
        np.save("cross_correlate_norm_{}_data.npy".format(norm),y)
    np.save("cross_correlate_t.npy".format(norm),x)
    
    viewer.show()
