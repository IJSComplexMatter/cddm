"""
Demonstrates how to compute fft of videos and the compute cross-correlation
function with the out-of-memory version of the multitau algorithm and do
live view of the computation.
"""
from cddm.viewer import CorrViewer
from cddm.video import multiply, normalize_video, crop, load, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.core import ccorr, normalize,stats
from cddm.multitau import log_average

import numpy as np
from examples.paper.conf import  PERIOD, SHAPE, KIMAX, KJMAX, NFRAMES_RANDOM, DATA_PATH

#: see video_simulator for details, loads sample video
import dual_video
import importlib
importlib.reload(dual_video) #recreates iterator

t1, t2 = dual_video.t1, dual_video.t2 

#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES_RANDOM

#:perform the actual multiplication
video = multiply(dual_video.video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#load in numpy array
fft1,fft2 = asarrays(fft, NFRAMES_RANDOM)

if __name__ == "__main__":
    import os.path as p

    #: now perform auto correlation calculation with default parameters 
    data = ccorr(fft1,fft2, t1 = t1,t2 = t2, n = NFRAMES_RANDOM)
    bg, var = stats(fft1,fft2)
    
    #: perform normalization and merge data
    data_lin = normalize(data, bg, var, scale = True)
    
    #: inspect the data
    viewer = CorrViewer(scale = True)
    viewer.set_data(data,bg, var)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
    
    #: change size, to define time resolution in log space
    x,y = log_average(data_lin, size = 16)
    
    #: save the normalized data to numpy files
    np.save(p.join(DATA_PATH, "corr_dual_t.npy"),x)
    np.save(p.join(DATA_PATH, "corr_dual_data.npy"),y)