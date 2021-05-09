""""Computes auto correlation of regular video and writes to disk for statistical analysis.
This is a long run!"""

# from cddm.viewer import MultitauViewer
from cddm.video import multiply, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, rfft2_crop
from cddm.sim import seed
from cddm.core import acorr, normalize, stats

import numpy as np
from examples.paper.simple_video.conf import SHAPE, NFRAMES_STANDARD, APPLY_WINDOW, DT_STANDARD
from examples.paper.conf import DATA_PATH, NRUN, KIMAX
# #: see video_simulator for details, loads sample video
import examples.paper.simple_video.standard_video as video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
if APPLY_WINDOW == False:
    window[...] = 1.
#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_STANDARD

#: determines optimal weight factor
from examples.paper.form_factor import g1, bg1,bg2
from cddm.norm import weight_from_g, weight_prime_from_g

bg1 = bg1(KIMAX,0)
bg2 = bg2(KIMAX,0)
delta = 0.

g1 = g1(np.arange(NFRAMES_STANDARD)*DT_STANDARD, KIMAX, 0)

w = weight_from_g(g1,delta)
wp = weight_prime_from_g(g1,delta, bg1,bg2)

def calculate():
    out = None
    bgs = []
    vars = []
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
        
        seed(i)
        importlib.reload(video_simulator) #recreates iterator with new seed
        
        
        video = multiply(video_simulator.video, window_video)
        
        fft = rfft2(video, kimax = 51, kjmax = 0)

        fft_array, = asarrays(fft,NFRAMES_STANDARD)
        
        data = acorr(fft_array)
        bg, var = stats(fft_array)

        bgs.append(bg)
        vars.append(var)
        for norm in (1,2,3,5,6,7,9,10,11):
            # weighted (subtracted)
            if norm in (7,11):
                y = normalize(data, bg, var, norm = norm, scale = True, weight = np.moveaxis(w,0,-1))
            # weighted prime (baseline)
            elif norm in (3,):
                y = normalize(data, bg, var, norm = norm, scale = True, weight = np.moveaxis(wp,0,-1))
            else:
                y = normalize(data, bg, var, norm = norm, scale = True)
            if out is None:
                out = np.empty(shape = (NRUN,12)+ y.shape, dtype = y.dtype)
            out[i,norm] = y
            
        
    return out, bgs, vars

if __name__ == "__main__":
    import os
    out, bgs, vars = calculate()
    bgs = np.asarray(bgs)
    vars = np.asarray(vars)
    np.save(os.path.join(DATA_PATH,"auto_standard_error_corr.npy"), out)
    np.save(os.path.join(DATA_PATH,"auto_standard_error_bg.npy"), bgs)
    np.save(os.path.join(DATA_PATH,"auto_standard_error_var.npy"), vars)

