"""
"""
from cddm.viewer import DataViewer
from cddm.video import multiply, asarrays
from cddm.window import blackman
from cddm.fft import rfft2
from cddm.core import acorr, normalize, stats
from cddm.multitau import log_average
from cddm.sim import seed
from cddm.norm import norm_flags

import numpy as np

#: see video_simulator for details, loads sample video

import importlib


import examples.paper.simple_video.standard_video as video_simulator

seed(0)
importlib.reload(video_simulator) #recreates iterator

from examples.paper.conf import KIMAX, KJMAX, SHAPE, APPLY_WINDOW, NFRAMES_STANDARD, \
 DATA_PATH, DT_STANDARD, NFRAMES


#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_STANDARD

#:perform the actual multiplication
if APPLY_WINDOW:
    video = multiply(video_simulator.video, window_video)
else:
    video = video_simulator.video


#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

#load in numpy array
fft_array, = asarrays(fft, NFRAMES_STANDARD)

if __name__ == "__main__":
    import os.path as p

    #: now perform auto correlation calculation with default parameters 
    data = acorr(fft_array,n = int(NFRAMES/DT_STANDARD), method = "fft")
    bg, var = stats(fft_array)
    
    for norm in (1,2,3,5,6,7,9,10,11):
    
        #: perform normalization and merge data
        data_lin = normalize(data, bg, var, scale = True, norm = norm)
        if norm == norm_flags(weighted = True, subtracted = True):
            np.save(p.join(DATA_PATH, "corr_standard_linear.npy"),data_lin)                
        #: perform log averaging
        x,y = log_average(data_lin, size = 16)
        
        #: save the normalized data to numpy files
        np.save(p.join(DATA_PATH, "corr_standard_t.npy"),x*DT_STANDARD)
        np.save(p.join(DATA_PATH, "corr_standard_data_norm{}.npy".format(norm)),y)

    #: inspect the data
    viewer = DataViewer()
    viewer.set_data(data_lin)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
