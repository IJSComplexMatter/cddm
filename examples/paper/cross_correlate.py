"""
"""
from cddm.viewer import CorrViewer
from cddm.video import multiply, normalize_video, crop, load, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.core import ccorr, normalize,stats
from cddm.multitau import log_average
from cddm.sim import seed

import importlib
import numpy as np
from examples.paper.conf import  PERIOD, SHAPE, KIMAX, KJMAX, NFRAMES_DUAL, DATA_PATH, APPLY_WINDOW

seed(0)

#: see video_simulator for details, loads sample video
import examples.paper.simple_video.dual_video as dual_video

importlib.reload(dual_video) #recreates iterator

t1, t2 = dual_video.t1, dual_video.t2 

#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES_DUAL

#:perform the actual multiplication
if APPLY_WINDOW:
    video = multiply(dual_video.video, window_video)
else:
    video = dual_video.video

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#load in numpy array
fft1,fft2 = asarrays(fft, NFRAMES_DUAL)

if __name__ == "__main__":
    import os.path as p

    #: now perform cross correlation calculation with default parameters 
    data = ccorr(fft1,fft2, t1 = t1,t2 = t2, n = NFRAMES_DUAL)
    bg, var = stats(fft1,fft2)
    
    for norm in range(8):
    
        #: perform normalization and merge data
        data_lin = normalize(data, bg, var, scale = True, norm = norm)

        if norm == 6:
            np.save(p.join(DATA_PATH, "corr_dual_linear.npy"),data_lin)

        #: change size, to define time resolution in log space
        x,y = log_average(data_lin, size = 16)
        
        #: save the normalized data to numpy files
        np.save(p.join(DATA_PATH, "corr_dual_t.npy"),x)
        np.save(p.join(DATA_PATH, "corr_dual_data_norm{}.npy".format(norm)),y)
    
     
    #: inspect the data
    viewer = CorrViewer(scale = True)
    viewer.set_data(data,bg, var)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
           
    
    