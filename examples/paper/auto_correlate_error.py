

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
from examples.paper.one_component.conf import NFRAMES, DT_RANDOM,NFRAMES_RANDOM, PERIOD_RANDOM, SHAPE, D, SIGMA, INTENSITY, NUM_PARTICLES, VMEAN, BACKGROUND, AREA_RATIO, APPLY_WINDOW

# #: see video_simulator for details, loads sample video
import examples.paper.one_component.random_video as video_simulator

from examples.paper.auto_correlate_random_error import window


data = np.load("auto_random_error_corr.npy")
bgs = np.load("auto_random_error_bg.npy")
vars = np.load("auto_random_error_var.npy")

data_regular = np.load("auto_fast_error_corr.npy")[...,0:NFRAMES]
    

#form factor, for relative signal intensity calculation
formf = rfft2_crop(form_factor(window, sigma = SIGMA, intensity = INTENSITY, dtype = "uint16", navg = 30), 51, 0)


#pp = auto_count_mixed(video_simulator.t, NFRAMES_RANDOM, PERIOD_RANDOM)
#pp[:,0]=1

scale = (np.abs(window)**2).sum() 


def g1(x,i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]**2 * AREA_RATIO
    #expected variance (signal + noise)
    #v = a + noise_level(window*video_simulator.dust,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    v = a + noise_level(window,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    #expected scalling factor
    a = a/v
    return a * np.exp(-D*(i**2+j**2)*x)

def fvar(i,j):
    #expected signal
    a = NUM_PARTICLES * formf[i,j]**2* AREA_RATIO
   # v = a + noise_level((window*video_simulator.dust1+window*video_simulator.dust2)/2,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5
    v = a + noise_level(window,BACKGROUND) #expected abs FFT squared of gaussian noise with std of BACKGROUND**0.5

    return v

BINNING = 1

LABELS = {0: "B'", 1 : "C'", 2 : "B", 3 : "C",4 : "W'", 6 : "W"}


MARKERS = {0: "1", 1 : "2", 2 : "3", 3 : "4",4 : "+", 6 : "x"}


plt.figure()

#estimated count for the random triggering experiment
n = random_time_count(NFRAMES_RANDOM, PERIOD_RANDOM)[0:NFRAMES_RANDOM]

#n = pp[:,0]


i,j = (16,0)

x = np.arange(NFRAMES_RANDOM)

#delta parameter for weight model.. it is zero by definition for auto-correlation
delta = 0.

var = vars.mean(0)

bg1 = (np.fft.rfft2(video_simulator.dust_frame*window)*BACKGROUND)[i,j] / (fvar(i,j))**0.5
bg2 = bg1

#bg1 = (np.fft.rfft2(window)*VMEAN)[i,j] / (fvar(i,j))**0.5
#bg2 = (np.fft.rfft2(window)*VMEAN)[i,j] / (fvar(i,j))**0.5


g = g1(x,i,j)

wp = weight_prime_from_g(g,delta,bg1,bg2)
w = weight_from_g(g,delta)

#error estimators using a simple model of independent data (delta = 0).

err0 = sigma_prime_weighted(0., g,  delta,bg1,bg2)/n**0.5
err1 = sigma_prime_weighted(1., g,  delta,bg1,bg2)/n**0.5
err2 = sigma_weighted(0., g,  delta)/n**0.5
err3 = sigma_weighted(1., g, delta)/n**0.5
err4 = sigma_prime_weighted(wp, g, delta,bg1,bg2)/n**0.5
err6 = sigma_weighted(w, g, delta)/n**0.5


w = weight_prime_from_g(g,delta,bg1,bg2)
err4 = sigma_prime_weighted(w, g, delta,bg1,bg2)/n**0.5
w = weight_prime_from_g(g,delta,0,0)
err6 = sigma_prime_weighted(w, g, delta,0,0)/n**0.5

#err0 = err0.mean(0)
#err1 = err1.mean(0)
#err4 = err4.mean(0)

ax1 = plt.subplot(121)
ax1.set_xscale("log")
ax1.set_xlabel(r"$\tau$")
ax1.set_title(r"$g(\tau),w(\tau)$ @ $q ={}$".format(i))

ax2 = plt.subplot(122)
ax2.set_title(r"$\sigma(\tau)$ @ $q ={}$".format(i))


x,y = merge_multilevel(multilevel(data_regular[:,2,i,j,:],binning = BINNING))
g = g1(x,i,j)
#g = y.mean(0)
std = (((y - g)**2).mean(axis = 0))**0.5
#ax1.semilogx(x[1:],y[:,1:].mean(0),marker = "o", linestyle = '',fillstyle = "none",label = "$g_R$", color = "k")
ax1.semilogx(x[1:],y[0,1:],marker = "o", linestyle = '',fillstyle = "none",label = "$g_R$", color = "k")

ax2.semilogx(x[1:],std[1:],marker = "o", linestyle = '', fillstyle = "none",label = "$\sigma_R$", color = "k")




for norm in (0,1,4,2,3,6):
    x,y = merge_multilevel(multilevel(data[:,norm,i,j,:],binning = BINNING))
    g = g1(x,i,j)
    #g = y.mean(0)
    std = (((y - g)**2).mean(axis = 0))**0.5
    ax1.semilogx(x[1:],y[0,1:],marker = MARKERS.get(norm,"o"), linestyle = '',fillstyle = "none",label = "$g_{}$".format(LABELS.get(norm)))
    #ax.semilogx(x,y.mean(0), "o",fillstyle = "none",label = "norm = {}".format(norm))

    ax2.semilogx(x[1:],std[1:],marker = MARKERS.get(norm,"o"), linestyle = '', fillstyle = "none",label = "$\sigma_{}$".format(LABELS.get(norm)))


ax1.plot(x[1:],g1(x[1:],i,j), "k",label = "$g$")

#: take first run, norm = 3 data for g estimation
x,g = log_average(data[0,3,i,j,:])
g = denoise(g)
g = decreasing(g)
g = g.clip(0,1)

#ax1.plot(x[1:],g[1:], "k:",label = "denoised")

x = np.arange(NFRAMES_RANDOM)
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

ax2.loglog(x[1:],err0[1:],"-")
ax2.loglog(x[1:],err1[1:],"-")
ax2.loglog(x[1:],err4[1:],"-")
ax2.loglog(x[1:],err2[1:],"-")
ax2.loglog(x[1:],err3[1:],"-")
ax2.loglog(x[1:],err6[1:],"-")


ax2.legend()
ax1.legend()




ax2.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"$\sigma$")
ax2.set_ylim(0.006,6)
ax1.set_ylabel(r"$g,w$")
ax1.set_ylim(-0.2,1.2)

plt.tight_layout()


