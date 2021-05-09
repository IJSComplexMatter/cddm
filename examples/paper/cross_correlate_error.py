""""Computes cross correlation of dual irregular-spaced video and writes 
computed data to disk"""

from cddm.video import multiply, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, rfft2_crop
from cddm.sim import seed
from cddm.core import ccorr, normalize, stats


import numpy as np
from examples.paper.conf import DATA_PATH, SHAPE,KIMAX,NFRAMES, APPLY_WINDOW, NRUN

#: see video_simulator for details, loads sample video
import examples.paper.simple_video.dual_video as video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)

if APPLY_WINDOW == False:
    #we still need window for form_factor calculation
    window[...] = 1.

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES


#: determines optimal weight factor
from examples.paper.form_factor import g1, bg1,bg2
from cddm.norm import weight_from_g, weight_prime_from_g

bg1 = bg1(KIMAX,0)
bg2 = bg2(KIMAX,0)
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
        
        fft = rfft2(video, kimax = KIMAX, kjmax = 0)

        f1, f2 = asarrays(fft,NFRAMES)
        bg, var = stats(f1,f2)
        
        bg, var = stats(f1,f2)
        data = ccorr(f1,f2, t1 = t1,t2=t2, n = NFRAMES)

    
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
    
    
    np.save(os.path.join(DATA_PATH,"cross_error_corr.npy"), out)
    np.save(os.path.join(DATA_PATH,"cross_error_bg.npy"), bgs)
    np.save(os.path.join(DATA_PATH,"cross_error_var.npy"), vars)
