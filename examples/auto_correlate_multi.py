"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""
from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iacorr_multi, normalize_multi, log_merge

import numpy as np

from examples.conf import NFRAMES, SHAPE, KIMAX, KJMAX, DATA_PATH
#: see video_simulator for details, loads sample video
import examples.video_simulator as video_simulator
import importlib
importlib.reload(video_simulator) #recreates iterator

#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES

#:perform the actual multiplication
video = multiply(video_simulator.video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

if __name__ == "__main__":
    import os.path as p

    #: now perform auto correlation calculation with default parameters using iterative algorithm
    data, bg, var = iacorr_multi(fft, count = NFRAMES)
    
    #: inspect the data
    viewer = MultitauViewer(scale = True)
    viewer.set_data(data, bg, var)
    viewer.set_mask(k = 25, angle = 0, sector = 30)
    viewer.plot()
    viewer.show()
    
    #perform normalization and merge data
    fast, slow = normalize_multi(data, bg, var, scale = True)
    
    #: save the normalized raw data to numpy files
    np.save(p.join(DATA_PATH,"auto_correlate_multi_raw_fast.npy"),fast)
    np.save(p.join(DATA_PATH,"auto_correlate_multi_raw_slow.npy"),slow)    
    
    x,y = log_merge(fast, slow)
    
    #: save the normalized merged data to numpy files
    np.save(p.join(DATA_PATH,"auto_correlate_multi_t.npy"),x)
    np.save(p.join(DATA_PATH,"auto_correlate_multi_data.npy"),y)


