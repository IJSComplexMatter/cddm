"""In this example we run the simulator of particle brownian motion
several times to calculate multiple correlation functions with different normalization
procedures. Then we plot the mean standard deviation of the data
points and compare that with the simple model error estimator.

This is a lengthy run... it will take a while to complete the experiment.
"""

# from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft, rfft2_crop
from cddm.sim import form_factor, seed, random_time_count
from cddm.core import ccorr, normalize, stats
from cddm.multitau import log_merge,  ccorr_multi_count, log_merge_count, multilevel, merge_multilevel, log_average
from cddm.norm import weight_from_data, sigma_prime_weighted, weight_prime_from_g, noise_delta, sigma_weighted, weight_from_g
from cddm.norm import noise_level
from cddm.avg import denoise,decreasing

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.conf import DATA_PATH
from examples.paper.simple_video.conf import KIMAX, NFRAMES, PERIOD, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, BACKGROUND, APPLY_WINDOW, AREA_RATIO, VMEAN

# #: see video_simulator for details, loads sample video
import examples.paper.simple_video.dual_video as video_simulator
import importlib

from examples.paper.form_factor import g1, bg1,bg2


#: create window for multiplication...
window = blackman(SHAPE)

if APPLY_WINDOW == False:
    #we still need window for form_factor calculation
    window[...] = 1.

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

#: how many runs to perform
NRUN = 16*2

bg1 = rfft2_crop(bg1(),KIMAX,0)
bg2 = rfft2_crop(bg2(),KIMAX,0)
delta = 0.

g1 = g1(np.arange(NFRAMES), KIMAX, 0)

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
        
        t1,t2 = video_simulator.t1,video_simulator.t2
        
        video = multiply(video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = KIMAX, kjmax = 0)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
        
        f1, f2 = asarrays(fft,NFRAMES)
        bg, var = stats(f1,f2)
        
        bg, var = stats(f1,f2)
        data = ccorr(f1,f2, t1 = t1,t2=t2, n = NFRAMES)

    
        #: now perform auto correlation calculation with default parameters and show live
        #data, bg, var = iacorr(fft, t,  auto_background = True, n = NFRAMES)
        #perform normalization and merge data
        bgs.append(bg)
        vars.append(var)
        
        #5 and 7 are redundand, but we are calulating it for easier indexing
        for norm in (1,2,3,5,6,7,9,10,11):
            # weighted (subtracted and compensated)
            if norm in (7,11):
                y = normalize(data, bg, var, norm = norm, scale = True, weight = np.moveaxis(w,0,-1))
            #weighted prime
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
    
    
    np.save(os.path.join(DATA_PATH,"cross_error_corr.npy"), out)
    np.save(os.path.join(DATA_PATH,"cross_error_bg.npy"), bgs)
    np.save(os.path.join(DATA_PATH,"cross_error_var.npy"), vars)
