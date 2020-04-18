"""
"""

from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge
import matplotlib.pyplot as plt

import numpy as np
from conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX, D

#: see video_simulator for details, loads sample video
import dual_video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

NRUN = 100

def calculate():
    out = None
    
    for i in range(NRUN):
        print("Run {}/{}".format(i+1,NRUN))
    
        importlib.reload(dual_video_simulator) #recreates iterator
        
        t1, t2 = dual_video_simulator.t1, dual_video_simulator.t2 
        
        #:perform the actual multiplication
        #video = multiply(dual_video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(dual_video_simulator.video, kimax = KIMAX, kjmax = KJMAX)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
    
    #    #we will show live calculation with the viewer
    #    viewer = MultitauViewer(scale = True)
    #    
    #    #initial mask parameters
    #    viewer.k = 15
    #    viewer.sector = 30
        
        #: now perform auto correlation calculation with default parameters and show live
        data, bg, var = iccorr_multi(fft, t1, t2, 
                 period = PERIOD)
        #perform normalization and merge data
        
        for norm in (0,1,2,3):
        
            fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
            x,y = log_merge(fast, slow)
        
            if out is None:
                out = np.empty(shape = (NRUN,4)+ y.shape, dtype = y.dtype)
                out[0,norm] = y
            else:
                out[i,norm] = y 
    return out
    
#out = calculate()

def g1(x,D,i,j):
    return np.exp(-D*(i**2+j**2)*x)

data = out.mean(axis = 0)
std = out.std(axis = 0)

i,j = (27,1)
y = g1(x,D,i,j)



ax = plt.subplot(121)
ax.set_xscale("log")
for norm in (2,3):
    std = (((out[:,:,i,j,:] - y)**2).mean(axis = 0))**0.5
    ax.errorbar(x,data[norm,i,j],std[norm]/(NRUN**0.5), fmt='.',label = "norm = {}".format(norm))



ax.plot(x,g1(x,D,i,j), "k",label = "true")

plt.subplot(122)
for norm in (2,3):
    std = (((out[:,:,i,j,:] - y)**2).mean(axis = 0))**0.5
    plt.semilogx(x,std[norm],label = "norm = {}".format(norm))

plt.legend()
