"""
"""

from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft, fft2_crop
from cddm.sim import form_factor
from cddm.multitau import iccorr_multi, normalize_multi, log_merge, count_multilevel, ccorr_multi_count, ccorr_multi_count2
import matplotlib.pyplot as plt
from cddm.conf import FDTYPE

import numpy as np
from examples.two_component.conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX, D1, D2, SIGMA1, SIGMA2, INTENSITY1,INTENSITY2

#: see video_simulator for details, loads sample video
import examples.two_component.dual_video_simulator as dual_video_simulator
import importlib

#: create window for multiplication...
window = blackman(SHAPE)
#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

NRUN = 40

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
        fft = rfft2(dual_video_simulator.video, kimax = 51, kjmax = 0)
        
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
        data, bg, var = iccorr_multi(fft, t1, t2, level_size = 16, binning = 0,
                 period = PERIOD)
        #perform normalization and merge data
        
        for norm in (0,1,2,3,4,5,6,7):
        
            fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
            x,y = log_merge(fast, slow)
        
            if out is None:
                out = np.empty(shape = (NRUN,8)+ y.shape, dtype = y.dtype)
                out[0,norm] = y
            else:
                out[i,norm] = y 
        
    return x, out, fast.shape, slow.shape

try:
    out
except NameError:
    x,out, fast_shape, slow_shape = calculate()


#compute form factors, for relative signal amplitudes
formf1 = fft2_crop(form_factor(SHAPE, sigma = SIGMA1, intensity = INTENSITY1), 51, 0)
formf2 = fft2_crop(form_factor(SHAPE, sigma = SIGMA2, intensity = INTENSITY2), 51, 0)

def g1(x,i,j):
    a = formf1[i,j]**2
    b = formf2[i,j]**2
    return a/(a+b)*np.exp(-D1*(i**2+j**2)*x)+b/(a+b)*np.exp(-D2*(i**2+j**2)*x)

x,t = ccorr_multi_count(NFRAMES, period = PERIOD, level_size = 16, binning = False)


err = 1/(t**0.5)

data = out.mean(axis = 0)
std = out.std(axis = 0)

i,j = (31,0)
y = g1(x,i,j)

err1 = err/2**0.5+ (1*y**4*err+0.*y**2*err)*(1- 1/2**0.5)

err1 = (1-y**2)*err/2**0.5+ y**2*err

err1 = ((1+y**2)/2.)**0.5 * err

err2 = err*(1-y)


errsum = (1/(1/err1**2 + 1/err2**2))**0.5



ax = plt.subplot(121)
ax.set_xscale("log")
plt.xlabel("delay time")
plt.title("Correlation @ k =({},{})".format(i,j))


for norm in (2,3,6,7):
    std = (((out[:,:,i,j,:] - y)**2).mean(axis = 0))**0.5
    #ax.errorbar(x,data[norm,i,j],std[norm]/(NRUN**0.5), fmt='.',label = "norm = {}".format(norm))
    ax.errorbar(x,out[0,norm,i,j],std[norm], fmt='.',label = "norm = {}".format(norm))

ax.plot(x,g1(x,i,j), "k",label = "true")

plt.legend()


plt.subplot(122)
plt.title("Mean error (std)")


plt.semilogx(x,err1,"k-")
plt.semilogx(x,err2,"k--")
#plt.semilogx(x,errsum[i,j],"k:")


for norm in (2,3,6,7):
    std = (((out[:,:,i,j,:] - y)**2).mean(axis = 0))**0.5
    plt.semilogx(x,std[norm],label = "norm = {}".format(norm))

#meanout = (out[:,2,:,:,:]/err1**2 + out[:,3,:,:,:]/err2**2)/(1/err1**2 + 1/err2**2) 

#std = (((meanout[:,i,j,:] - y)**2).mean(axis = 0))**0.5
#plt.semilogx(x,std,label = "mean".format(norm))
plt.xlabel("delay time")

plt.legend()
