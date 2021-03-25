""""Computes auto correlation of random video and writes to disk for statistical analysis.
This is a long run!"""

# from cddm.viewer import MultitauViewer
from cddm.video import multiply, asarrays
from cddm.window import blackman
from cddm.fft import rfft2
from cddm.sim import seed
from cddm.core import acorr, normalize, stats

import numpy as np
from examples.paper.simple_video.conf import APPLY_WINDOW, NFRAMES_RANDOM, SHAPE, NFRAMES, DT_RANDOM
from examples.paper.conf import DATA_PATH, NRUN
# #: see video_simulator for details, loads sample video
import examples.paper.simple_video.random_video as video_simulator
import importlib, os

#: create window for multiplication...
window = blackman(SHAPE)
if APPLY_WINDOW == False:
    window[...] = 1.
#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_RANDOM

def calculate():
    out = None
    bgs = []
    vars = []
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
        
        seed(i)
        importlib.reload(video_simulator) #recreates iterator with new seed
        
        t = video_simulator.t
        
        video = multiply(video_simulator.video, window_video)

        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)

        fft_array, = asarrays(fft,NFRAMES_RANDOM)
        
        data = acorr(fft_array, t = t, n = int(NFRAMES/DT_RANDOM))
        bg, var = stats(fft_array)

        bgs.append(bg)
        vars.append(var)
        for norm in (1,2,3,5,6,7,9,10,11):
            y = normalize(data, bg, var, norm = norm, scale = True)
            if out is None:
                out = np.empty(shape = (NRUN,12)+ y.shape, dtype = y.dtype)
            out[i,norm] = y
             
    return out, bgs, vars

if __name__ == "__main__":
    out, bgs, vars = calculate()
    bgs = np.asarray(bgs)
    vars = np.asarray(vars)
    np.save(os.path.join(DATA_PATH,"auto_random_error_corr.npy"), out)
    np.save(os.path.join(DATA_PATH,"auto_random_error_bg.npy"), bgs)
    np.save(os.path.join(DATA_PATH,"auto_random_error_var.npy"), vars)
