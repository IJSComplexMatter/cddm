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
from cddm.core import acorr, normalize, stats
from cddm.multitau import log_merge,  ccorr_multi_count, log_merge_count, multilevel, merge_multilevel, log_average
from cddm.norm import weight_from_data,weight_from_g, noise_level
from cddm.avg import denoise,decreasing
from cddm._core_nb import sigma_weighted_auto_general, auto_count_mixed, sigma_weighted

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.conf import NFRAMES_RANDOM, PERIOD, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, VMEAN, BACKGROUND

# #: see video_simulator for details, loads sample video
import examples.paper.random_video as video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
window[...] = 1.
#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_RANDOM

#: how many runs to perform
NRUN = 4*16

def calculate():
    out = None
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
        
        seed(i)
        importlib.reload(video_simulator) #recreates iterator with new seed
        
        t = video_simulator.t
        
        video = multiply(video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
        
        fft_array, = asarrays(fft,NFRAMES_RANDOM)
        
        data = acorr(fft_array, t = t, n = NFRAMES_RANDOM)
        bg, var = stats(fft_array)
    
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
formf = rfft2_crop(form_factor(window, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16", navg = 1000), 51, 0)


pp = auto_count_mixed(video_simulator.t, NFRAMES_RANDOM, PERIOD)
#pp[:,0]=1

scale = (np.abs(window)**2).sum() 


def g1(x,i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]#*0.8858 
    #expected variance (signal + noise)
    v = a + noise_level(window,BACKGROUND)
    #v = a + SHAPE[0]*SHAPE[1]*BACKGROUND#expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    #expected scalling factor
    a = a/v
    return a * np.exp(-D*(i**2+j**2)*x)

LABELS = {2 : "subtracted", 3 : "compensated", 6 : "weighted"}

plt.figure()

#estimated count for the random triggering experiment
n = random_time_count(NFRAMES_RANDOM, PERIOD)[0:NFRAMES_RANDOM]

data = out

i,j = (26,0)

x = np.arange(NFRAMES_RANDOM)

#delta parameter for weight model.. it is zero by definition for auto-correlation
delta = 0.
noise = noise_level(window,BACKGROUND)


#expected signal
a =NUM_PARTICLES * formf[i,j] #*0.8858 
#expected variance (signal + noise)
noise = noise_level(window,BACKGROUND)
#expected scalling factor
noise = noise/(noise + a)

g = g1(x,i,j)
w = weight_from_g(g,noise,delta)

#error estimators using a simple model of independent data (delta = 0).
err2 = sigma_weighted(0., g, noise, delta)/n**0.5
err3 = sigma_weighted(1., g, noise, delta)/n**0.5
err6 = sigma_weighted(w, g, noise, delta)/n**0.5

w[...]=0
err6  = sigma_weighted_auto_general(w,g,(noise,)*NFRAMES_RANDOM,pp)/n**0.5
#err3  = sigma_weighted_auto_general(w*0.,g,(noise,)*NFRAMES_RANDOM,pp)/n**0.5
#err2  = sigma_weighted_auto_general(w*0+1,g,(noise,)*NFRAMES_RANDOM,pp)/n**0.5
err6  = sigma_weighted(1., g, 0, delta)/n**0.5

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
    std = ((((y - g)**2)).mean(axis = 0))**0.5
    ax1.semilogx(x[1:],y[:,1:].mean(0),fillstyle = "none",label = "{}".format(LABELS.get(norm)))
    #ax.semilogx(x,y.mean(0), "o",fillstyle = "none",label = "norm = {}".format(norm))

    ax2.semilogx(x[1:],std[1:],label = "{}".format(LABELS.get(norm)))


ax1.plot(x[1:],g1(x[1:],i,j), "k",label = "model")


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


