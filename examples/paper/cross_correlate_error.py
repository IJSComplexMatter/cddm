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
from cddm.norm import weight_from_data, sigma_weighted, weight_from_g, noise_delta
from cddm.avg import denoise,decreasing

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.conf import NFRAMES, PERIOD, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, BACKGROUND

# #: see video_simulator for details, loads sample video
import examples.paper.dual_video as video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

#: how many runs to perform
NRUN = 16

def calculate():
    out = None
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
        
        seed(i)
        importlib.reload(video_simulator) #recreates iterator with new seed
        
        t1,t2 = video_simulator.t1,video_simulator.t2
        
        video = multiply(video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)
        
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
        
        #5 and 7 are redundand, but we are calulating it for easier indexing
        for norm in (0,1,2,3,4,5,6,7):
            y = normalize(data, bg, var, norm = norm, scale = True)
            if out is None:
                out = np.empty(shape = (NRUN,8)+ y.shape, dtype = y.dtype)
            out[i,norm] = y
        
    return out

try:
    out
except NameError:
    out = calculate()


#form factor, for relative signal intensity calculation
formf = rfft2_crop(form_factor(SHAPE, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16"), 51, 0)

def g1(x,i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]**2
    #expected variance (signal + noise)
    v = a + SHAPE[0]*SHAPE[1]*BACKGROUND #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    #expected scalling factor
    a = a/v
    return a * np.exp(-D*(i**2+j**2)*x)

LABELS = {2 : "subtracted", 3 : "compensated", 6 : "weighted"}

plt.figure()

#estimated count for the random triggering experiment
n = NFRAMES/PERIOD*2

data = out

i,j = (8,0)

x = np.arange(NFRAMES)

#delta parameter for weight model.. it is zero by definition for auto-correlation
delta = 0.
noise = 1-g1(0,i,j)

g = g1(x,i,j)
w = weight_from_g(g,noise,delta)

#error estimators using a simple model of independent data (delta = 0).
err2 = sigma_weighted(1., g, noise,delta)/n**0.5
err3 = sigma_weighted(0., g, noise,delta)/n**0.5
err6 = sigma_weighted(w, g, noise,delta)/n**0.5

ax1 = plt.subplot(121)
ax1.set_xscale("log")
ax1.set_xlabel("delay time")
ax1.set_title("$g$ @ $k =({},{})$".format(i,j))

ax2 = plt.subplot(122)
ax2.set_title("$\sigma$ @ $k =({},{})$".format(i,j))


for norm in (2,3,6):
    x,y = merge_multilevel(multilevel(data[:,norm,i,j,:],binning = 0))
    g = g1(x,i,j)
    #g = y.mean(0)
    std = (((y - g)**2).mean(axis = 0))**0.5
    ax1.semilogx(x[1:],y[0,1:],fillstyle = "none",label = "{}".format(LABELS.get(norm)))
    #ax.semilogx(x,y.mean(0), "o",fillstyle = "none",label = "norm = {}".format(norm))

    ax2.semilogx(x[1:],std[1:],label = "{}".format(LABELS.get(norm)))


ax1.plot(x[1:],g1(x[1:],i,j), "k",label = "model")

#: take first run, norm = 3 data for g estimation
x,g = log_average(data[0,3,i,j,:])
g = denoise(g)
g = decreasing(g)
g = g.clip(0,1)

ax1.plot(x[1:],g[1:], "k:",label = "denoised")


x,err2 = merge_multilevel(multilevel(err2,binning = 0))
x,err3 = merge_multilevel(multilevel(err3,binning = 0))
x,err6 = merge_multilevel(multilevel(err6,binning = 0))


ax2.semilogx(x[1:],err2[1:],"k:", label = "$\sigma_S$ (expected)")
ax2.semilogx(x[1:],err3[1:],"k--", label = "$\sigma_C$ (expected)")
ax2.semilogx(x[1:],err6[1:],"k-", label = "$\sigma_W$ (expected)")

ax2.set_xlabel("delay time")

ax2.legend()
ax1.legend()


