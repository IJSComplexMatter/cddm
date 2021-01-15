

# from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft, rfft2_crop
from cddm.sim import form_factor, seed, random_time_count
from cddm.core import acorr, normalize, stats
from cddm.multitau import log_merge,  ccorr_multi_count, log_merge_count, multilevel, merge_multilevel, log_average
from cddm.norm import weight_from_data,weight_from_g, noise_level, weight_prime_from_g
from cddm.avg import denoise,decreasing
from cddm._core_nb import  auto_count_mixed, sigma_weighted, sigma_prime_weighted

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.simple_video.conf import NFRAMES, DT_RANDOM,NFRAMES_RANDOM, PERIOD_RANDOM, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, VMEAN, BACKGROUND, AREA_RATIO, APPLY_WINDOW
from examples.paper.conf import DATA_PATH
# #: see video_simulator for details, loads sample video
import examples.paper.simple_video.fast_video as video_simulator
import importlib, os

#: create window for multiplication...
window = blackman(SHAPE)
if APPLY_WINDOW == False:
    window[...] = 1.
#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_RANDOM

#: how many runs to perform
NRUN = 16*2

def calculate():
    out = None
    bgs = []
    vars = []
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
        
        seed(i)
        importlib.reload(video_simulator) #recreates iterator with new seed
        
        
        video = multiply(video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
        
        fft_array, = asarrays(fft,NFRAMES_RANDOM)
        
        data = acorr(fft_array)
        bg, var = stats(fft_array)
    
        #: now perform auto correlation calculation with default parameters and show live
        #data, bg, var = iacorr(fft, t,  auto_background = True, n = NFRAMES)
        #perform normalization and merge data
        
        #5 and 7 are redundand, but we are calulating it for easier indexing
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
    np.save(os.path.join(DATA_PATH,"auto_fast_error_corr.npy"), out)
    np.save(os.path.join(DATA_PATH,"auto_fast_error_bg.npy"), bgs)
    np.save(os.path.join(DATA_PATH,"auto_fast_error_var.npy"), vars)

