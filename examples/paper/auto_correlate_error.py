"""In this example we run the simulator of single-particle brownian motion
several times to calculate multiple correlation functions with different normalization
procedures and binning = 1 and binning = 0. Then we plot the mean standard deviation of the data
points and compare that with the simple model error estimator.

This is a lengthy run... it will take a while to complete the experiment.
"""

from cddm.video import multiply, normalize_video, crop, asarrays
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft, rfft2_crop
from cddm.sim import seed
from cddm.core import acorr, normalize, stats
import matplotlib.pyplot as plt
from cddm.conf import FDTYPE
from cddm.multitau import merge_multilevel, multilevel, multilevel_count

import numpy as np
from examples.paper.conf import NFRAMES, PERIOD, SHAPE,  D

#: see video_simulator for details, loads sample video
import examples.paper.video_simulator as video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES

NRUN = 8*16

def calculate():
    out = None
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
    
        importlib.reload(video_simulator) #recreates iterator
        
        #reset seed... because we use seed(0) in dual_video_simulator
        seed(i)
        
        t = video_simulator.t
        
        video = multiply(video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
        fft_array, = asarrays(fft,NFRAMES)
        #: now perform auto correlation calculation with default parameters and show live
        bg, var = stats(fft_array)
        data = acorr(fft_array, t, n = NFRAMES)
        #perform normalization and merge data
        
        #5 and 7 are redundand, but we are calulating it for easier indexing
        for norm in (0,1,2,3,4,5,6,7):
        
            y = normalize(data, bg, var, norm = norm, scale = True)

            if out is None:
                out = np.empty(shape = (NRUN,8)+ y.shape, dtype = y.dtype)
                out[0,norm] = y
            else:
                out[i,norm] = y 
        
    return out, data[1] 

try:
    out
except NameError:
    out, count = calculate()

def g1(x,i,j):
    return np.exp(-D*(i**2+j**2)*x)

data = out

plt.figure()

n = np.ones(NFRAMES)*32
n[1:32] = np.arange(32)[1:32]
n[0] = NFRAMES

#n = count
#get eefective count in aveariging... 

x = np.arange(NFRAMES)

y = multilevel(data, level_size = 16, binning = 1)
n = multilevel_count(n, level_size = 16,binning = 1)
x,data = merge_multilevel(y)
x,n = merge_multilevel(n)



i,j = (16,0)

y = g1(x,i,j)

#error estimators using a simple model of independent data.
err2 = ((1+y**2)/2./n)**0.5 
err3 = (((1-y)**2)/n)**0.5
err6 = (0.5*(y**2-1)**2 / (y**2+1)/n)**0.5

ax = plt.subplot(121)
ax.set_xscale("log")
plt.xlabel("delay time")
plt.title("Correlation @ k =({},{})".format(i,j))


for norm in (2,3,6):
    std = (((data[:,:,i,j,:] - y)**2).mean(axis = 0))**0.5
    ax.semilogx(x,data[0,norm,i,j], label = "norm = {}".format(norm))

ax.plot(x,g1(x,i,j), "k",label = "true")

plt.legend()


plt.subplot(122)
plt.title("Mean error (std)")

for norm in (2,3,6):
    std = np.nanmean(((data[:,:,i,j,:] - y)**2),axis = 0)**0.5
    plt.semilogx(x,std[norm],label = "norm = {}".format(norm))

plt.semilogx(x,err2,"k:", label = "norm 2 (expected)")
plt.semilogx(x,err3,"k--", label = "norm 3 (expected)")
plt.semilogx(x,err6,"k-", label = "norm 6 (expected)")

plt.xlabel("delay time")

plt.legend()

