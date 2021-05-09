"""
"""
from cddm.viewer import  CorrViewer
from cddm.video import multiply,  asarrays
from cddm.window import blackman
from cddm.fft import rfft2
from cddm.core import acorr, normalize, stats
from cddm.multitau import log_average
from cddm.sim import seed
import importlib

import numpy as np

seed(0)

#: see video_simulator for details, loads sample video
import examples.paper.simple_video.random_video as video_simulator


importlib.reload(video_simulator) #recreates iterator

from examples.paper.conf import KIMAX, KJMAX, SHAPE, NFRAMES_RANDOM, NFRAMES, DATA_PATH, APPLY_WINDOW, DT_RANDOM


#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_RANDOM

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
fft_array, = asarrays(fft, NFRAMES_RANDOM)

if __name__ == "__main__":
    import os.path as p

    #: now perform auto correlation calculation with default parameters 
    data = acorr(fft_array, t = video_simulator.t, n = int(NFRAMES/DT_RANDOM))
    bg, var = stats(fft_array)
    
    for norm in (1,2,3,5,6,7,9,10,11):
    
        #: perform normalization and merge data
        data_lin = normalize(data, bg, var, scale = True, norm = norm)
    
        #: change size, to define time resolution in log space
        x,y = log_average(data_lin, size = 16)
        
        #: save the normalized data to numpy files
        np.save(p.join(DATA_PATH, "corr_random_t.npy"),x*DT_RANDOM)
        np.save(p.join(DATA_PATH, "corr_random_data_norm{}.npy".format(norm)),y)

    
    #: inspect the data
    viewer = CorrViewer(scale = True)
    viewer.set_data(data,bg, var)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
    
