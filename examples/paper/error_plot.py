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
from cddm.multitau import log_merge,  ccorr_multi_count, acorr_multi_count,log_merge_count, multilevel, merge_multilevel, log_average
from cddm.norm import weight_from_data, sigma_prime_weighted, weight_prime_from_g, noise_delta, sigma_weighted, weight_from_g
from cddm.norm import noise_level
from cddm.avg import denoise,decreasing

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.simple_video.conf import NFRAMES, NFRAMES_RANDOM, PERIOD_RANDOM, PERIOD, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, BACKGROUND, APPLY_WINDOW, AREA_RATIO, VMEAN, ADC_SCALE_FACTOR

# #: see video_simulator for details, loads sample video
import examples.paper.simple_video.dual_video as video_simulator
import importlib
from examples.paper.cross_correlate_error import window
from examples.paper.conf import DATA_PATH
from examples.paper.conf import SAVE_FIGS
import os


CROSS = False

BINNING_DATA = 1

BINNING_ERROR = 0

BINNING_MODEL = 0


K = 16

if CROSS:

    data = np.load(os.path.join(DATA_PATH,"cross_error_corr.npy"))
    bgs = np.load(os.path.join(DATA_PATH,"cross_error_bg.npy"))
    vars = np.load(os.path.join(DATA_PATH,"cross_error_var.npy"))

    data_regular = np.load(os.path.join(DATA_PATH,"auto_standard_error_corr.npy"))[...,0:NFRAMES//PERIOD*2]
    #data_regular = data
else:
    data = np.load(os.path.join(DATA_PATH,"auto_random_error_corr.npy"))
    bgs = np.load(os.path.join(DATA_PATH,"auto_random_error_bg.npy"))
    vars = np.load(os.path.join(DATA_PATH,"auto_random_error_var.npy"))

    data_regular = np.load(os.path.join(DATA_PATH,"auto_fast_error_corr.npy"))[...,0:NFRAMES]


#form factor, for relative signal intensity calculation
formf = rfft2_crop(form_factor(window, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16", navg = 30), 51, 0)
#formf2 = rfft2_crop(form_factor(window*video_simulator.dust2, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16", navg = 200), 51, 0)

def g1(x,i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]**2 * AREA_RATIO
    #expected variance (signal + noise)
    if CROSS:
        v = a + noise_level((window*video_simulator.dust1+window*video_simulator.dust2)/2,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    else:
        v = a + noise_level(window*video_simulator.dust1,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
        
    
    #v = a + noise_level(window,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    #expected scalling factor
    a = a/v
    return a * np.exp(-D*(i**2+j**2)*x)

def fvar(i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]**2 * AREA_RATIO
    if CROSS:
        v = a + noise_level((window*video_simulator.dust1+window*video_simulator.dust2)/2,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    else:
        v = a + noise_level(window*video_simulator.dust1,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    #v = a + noise_level(window,BACKGROUND)#expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    return v/(ADC_SCALE_FACTOR**2)

def fdelta(i,j):
    a = NUM_PARTICLES * formf[i,j]**2 * AREA_RATIO
    n = noise_level(window,BACKGROUND)#expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    return n/(a+n)


LABELS = {1: "B'", 2 : "S'", 3 : "W'", 5 : "B", 6 : "S", 7 : "W", 9 : "B''", 10 : "S''", 11 : "W''"  }


MARKERS = {1: "1", 2 : "2", 3 : "3", 5 : "4",6 : "+", 7 : "x", 9 : "4", 10 : "+", 11 : "x"}


plt.figure()



if not BINNING_MODEL:
    
    #estimated count for the random triggering experiment
    if CROSS:
        n = NFRAMES/PERIOD*2
    else:
        n = random_time_count(NFRAMES_RANDOM, PERIOD_RANDOM)[0:NFRAMES_RANDOM]
else: 
    if CROSS:
        clin,cmulti = ccorr_multi_count(NFRAMES, period = PERIOD, level_size = 16, binning = 1)
    else:
        clin,cmulti = acorr_multi_count(NFRAMES_RANDOM, period = PERIOD_RANDOM, level_size = 16, binning = 1)
    
   #get eefective count in aveariging... 
    x,n = log_merge_count(clin, cmulti, binning = 1)




i,j = (K,0)

x = np.arange(NFRAMES)

#delta parameter for weight model.. it 
delta = 0.

# var = vars[:,0]* 0.5 + vars[:,1] * 0.5 
# var = var.mean(0)

# delta = vars[:,1]* 0.5 - vars[:,0] * 0.5 
# delta = delta.mean(0)
# delta = delta[i,j]/var[i,j]

#def fvar(i,j):
#    return var[i,j]

bg1 = (np.fft.rfft2(video_simulator.dust1*window)*VMEAN)[i,j] / (fvar(i,j))**0.5
bg2 = (np.fft.rfft2(video_simulator.dust2*window)*VMEAN)[i,j] / (fvar(i,j))**0.5 if CROSS else bg1



#bg1 = (np.fft.rfft2(window)*VMEAN)[i,j] / (fvar(i,j))**0.5
#bg2 = (np.fft.rfft2(window)*VMEAN)[i,j] / (fvar(i,j))**0.5



# bg1,bg2 = bgs.mean(0)

# bg1 = bg1[i,j]
# bg2 = bg2[i,j]

# bg1 = bg1/(fvar(i,j))**0.5
# bg2 = bg2/(fvar(i,j))**0.5

#bg1 = np.fft.rfft2(window*BACKGROUND/2/2/2)[i,j] / (fvar(i,j))**0.5
#bg2 = np.fft.rfft2(window*BACKGROUND/2/2/2)[i,j] / (fvar(i,j))**0.5


# bg1 = bgs[:,0,i,j]**2 / (var[i,j])
# bg2 = bgs[:,1,i,j]**2 / (var[i,j])

# bg1 = bg1[...,None]
# bg2 = bg2[...,None]

# bg1 = bg1.mean(0)
# bg2 = bg2.mean(0)

# bg1 = bg1**0.5
# bg2 = bg2**0.5



g = g1(x,i,j)
wp = weight_prime_from_g(g,delta,bg1,bg2)
w = weight_from_g(g, delta)

#error estimators using a simple model of independent data (delta = 0).

err0 = sigma_prime_weighted(0., g+0j,  delta,bg1,bg2)#/n**0.5
err1 = sigma_prime_weighted(1., g,  delta,bg1,bg2)#/n**0.5
err2 = sigma_weighted(0., g,  delta)#/n**0.5
err3 = sigma_weighted(1., g, delta)#/n**0.5
err4 = sigma_prime_weighted(wp, g, delta,bg1,bg2)#/n**0.5
err6 = sigma_weighted(w, g, delta)#/n**0.5

err4 = sigma_prime_weighted(wp, g, delta,bg1,bg2)#/n**0.5
err6 = sigma_prime_weighted(w, g, delta,0,0)#/n**0.5

#err0 = err0.mean(0)
#err1 = err1.mean(0)
#err4 = err4.mean(0)

ax1 = plt.subplot(121)
ax1.set_xscale("log")
ax1.set_xlabel(r"$\tau$")
ax1.set_title(r"$g(\tau), w(\tau)$ @ $q = {}$".format(K))

ax2 = plt.subplot(122)
ax2.set_title(r"$\sigma (\tau)$ @ $q = {}$".format(K))

for binning in (0,1):
    x,y = merge_multilevel(multilevel(data_regular[:,2,i,j,:],binning = binning))
    if CROSS:
        x = x*PERIOD//2
    g = g1(x,i,j)
    #g = y.mean(0)
    std = (((y - g)**2).mean(axis = 0))**0.5
    #ax1.semilogx(x[1:],y[:,1:].mean(0),marker = "o", linestyle = '',fillstyle = "none",label = "$g_R$", color = "k")
    if binning == BINNING_DATA:
        ax1.semilogx(x[1:],y[0,1:],marker = "o", linestyle = '',fillstyle = "none",label = "$R$", color = "k")
        
    if binning == BINNING_ERROR:
        ax2.semilogx(x[1:],std[1:],marker = "o", linestyle = '', fillstyle = "none",label = "$R$", color = "k")
    else:
        #ax1.semilogx(x[1:],y[0,1:],linestyle = ':',fillstyle = "none", color = "k")

        ax2.semilogx(x[1:],std[1:],linestyle = ':', fillstyle = "none", color = "k")




for binning in (0,1):
    ax1.set_prop_cycle(None)
    ax2.set_prop_cycle(None)
    for norm in (1,2,3,5,6,7):
    
        x,y = merge_multilevel(multilevel(data[:,norm,i,j,:],binning = binning))
        g = g1(x,i,j)
        #g = y.mean(0)
        std = (((y - g)**2).mean(axis = 0))**0.5
        #ax1.semilogx(x[1:],y[:,1:].mean(0),marker = MARKERS.get(norm,"o"), linestyle = '',fillstyle = "none",label = "$g_{}$".format(LABELS.get(norm)))
        
        if binning == BINNING_DATA:
            ax1.semilogx(x[1:],y[0,1:],marker = MARKERS.get(norm,"o"), linestyle = '',fillstyle = "none",label = "${}$".format(LABELS.get(norm)))
        if binning == BINNING_ERROR:
            ax2.semilogx(x[1:],std[1:],marker = MARKERS.get(norm,"o"), linestyle = '', fillstyle = "none",label = "${}$".format(LABELS.get(norm)))
        else:
            #ax1.semilogx(x[1:],y[0,1:],linestyle = ':',fillstyle = "none")
    
            ax2.semilogx(x[1:],std[1:],linestyle = ':', fillstyle = "none")

            





ax1.plot(x[1:],g1(x[1:],i,j), "k",label = "$g$")

#: take first run, norm = 3 data for g estimation
x,g = log_average(data[0,3,i,j,:])
g = denoise(g)
g = decreasing(g)
g = g.clip(0,1)

#ax1.plot(x[1:],g[1:], "k:",label = "denoised")

x = np.arange(NFRAMES)
ax1.plot(x[1:],w[1:], "k--",label = "$w$")
ax1.plot(x[1:],wp[1:], "k:",label = "$w'$")

#ax2.set_ylim(ax1.get_ylim())


x,err2 = merge_multilevel(multilevel(err2,binning = 0))
x,err3 = merge_multilevel(multilevel(err3,binning = 0))
x,err6 = merge_multilevel(multilevel(err6,binning = 0))
x,err0 = merge_multilevel(multilevel(err0,binning = 0))
x,err1 = merge_multilevel(multilevel(err1,binning = 0))
x,err4 = merge_multilevel(multilevel(err4,binning = 0))

ax2.set_prop_cycle(None)


nmax = len(x)
if BINNING_MODEL or not CROSS:
    n = n[1:nmax]

ax2.loglog(x[1:],err0[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err1[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err4[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err2[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err3[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err6[1:]/np.sqrt(n),"-")

ax2.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"$\sigma$")
ax2.set_ylim(0.001,2)
ax1.set_ylabel(r"$g,w$")
ax1.set_ylim(-1,1.5)

#ax2.legend()
ax1.legend(loc = 3)

plt.tight_layout()

if SAVE_FIGS:
    if CROSS:
        plt.savefig("plots/plot_cross_error_{}.pdf".format(K))
    else:
        plt.savefig("plots/plot_auto_error_{}.pdf".format(K))
        
plt.show()


