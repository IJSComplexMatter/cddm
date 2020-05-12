"""In this example we run the simulator of two-component particle brownian motion
several times to calculate multiple correlation functions with different normalization
procedures and binning = 1 and binning = 0. Then we plot the mean standard deviation of the data
points and compare that with the simple model error estimator.

This is a lengthy run... it will take a while to complete the experiment.
"""

# from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft, rfft2_crop
from cddm.sim import form_factor, seed
from cddm.multitau import iccorr_multi, normalize_multi, log_merge,  ccorr_multi_count, log_merge_count
import matplotlib.pyplot as plt

import numpy as np
from examples.two_component.conf import NFRAMES, PERIOD, SHAPE, D1, D2, SIGMA1, SIGMA2, INTENSITY1,INTENSITY2

# #: see video_simulator for details, loads sample video
import examples.two_component.dual_video_simulator as dual_video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

NRUN = 8*8

def calculate(binning = 1):
    out = None
    
    for i in range(NRUN):
        
        print("Run {}/{}".format(i+1,NRUN))
    
        importlib.reload(dual_video_simulator) #recreates iterator
        
        #reset seed... because we use seed(0) in dual_video_simulator
        seed(i)
        
        t1, t2 = dual_video_simulator.t1, dual_video_simulator.t2 
        
        video = multiply(dual_video_simulator.video, window_video)
        
        #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
        #video = normalize_video(video)
        
        #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
        fft = rfft2(video, kimax = 51, kjmax = 0)
        
        #: you can also normalize each frame with respect to the [0,0] component of the fft
        #: this it therefore equivalent to  normalize_video
        #fft = normalize_fft(fft)
    
        #: now perform auto correlation calculation with default parameters and show live
        data, bg, var = iccorr_multi(fft, t1, t2, level_size = 16, binning = binning,
                 period = PERIOD, auto_background = True)
        #perform normalization and merge data
        
        #5 and 7 are redundand, but we are calulating it for easier indexing
        for norm in (0,1,2,3,4,5,6,7):
        
            fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
            
            #we merge with binning (averaging) of linear data enabled/disabled
            x,y = log_merge(fast, slow, binning = binning)
        
            if out is None:
                out = np.empty(shape = (NRUN,8)+ y.shape, dtype = y.dtype)
                out[0,norm] = y
            else:
                out[i,norm] = y 
        
    return x, out

try:
    out
except NameError:
    x,data_0 = calculate(0)
    #x,data_1 = calculate(1)
    out = [data_0, data_0]


#compute form factors, for relative signal amplitudes
formf1 = rfft2_crop(form_factor(SHAPE, sigma = SIGMA1, intensity = INTENSITY1, dtype = "uint16"), 51, 0)
formf2 = rfft2_crop(form_factor(SHAPE, sigma = SIGMA2, intensity = INTENSITY2, dtype = "uint16"), 51, 0)

def g1(x,i,j):
    a = formf1[i,j]**2
    b = formf2[i,j]**2
    return a/(a+b)*np.exp(-D1*(i**2+j**2)*x)+b/(a+b)*np.exp(-D2*(i**2+j**2)*x)


for binning in (0,):
    plt.figure()

    clin,cmulti = ccorr_multi_count(NFRAMES, period = PERIOD, level_size = 16, binning = binning)
    
    #get eefective count in aveariging... 
    x,n = log_merge_count(clin, cmulti, binning = binning)
    data = out[binning]
    
    i,j = (21,0)
    
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
        std = (((data[:,norm,i,j,:] - y)**2).mean(axis = 0))**0.5
        ax.errorbar(x,data[:,norm,i,j].mean(0),std, fmt='.',label = "norm = {}".format(norm))
    
    ax.plot(x,g1(x,i,j), "k",label = "true")
    
    plt.legend()
    
    
    plt.subplot(122)
    plt.title("Mean error (std)")
    
    for norm in (2,3,6):
        y = data[:,norm,i,j,:].mean(0)
        std = (((data[:,norm,i,j,:] - y)**2).mean(axis = 0))**0.5
        plt.semilogx(x,std,label = "norm = {}".format(norm))
    
    plt.semilogx(x,err2,"k:", label = "norm 2 (expected)")
    plt.semilogx(x,err3,"k--", label = "norm 3 (expected)")
    plt.semilogx(x,err6,"k-", label = "norm 6 (expected)")
    
    plt.xlabel("delay time")
    
    plt.legend()





