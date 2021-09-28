"""Plots fig 3. and fig 4. from the paper.

You must first create data calling the following scripts:
    
$ python auto_correlate_random_error.py
$ python auto_correlate_standard_error.py
$ python auto_correlate_fast_error.py
$ python cross_correlate_error.py

"""

from cddm.sim import random_time_count
from cddm.multitau import ccorr_multi_count, acorr_multi_count,log_merge_count, multilevel, merge_multilevel
from cddm.norm import sigma_prime_weighted, weight_prime_from_g, sigma_weighted, weight_from_g
#from cddm.avg import denoise,decreasing

import matplotlib.pyplot as plt

import numpy as np
from examples.paper.conf import NFRAMES, PERIOD , NFRAMES_RANDOM, PERIOD_RANDOM

from examples.paper.conf import DATA_PATH
from examples.paper.conf import SAVE_FIGS
import os

from examples.paper.form_factor import g1, bg1, bg2

#: whether toplot cross-correlationor auto-correlation data
CROSS = False

#: whether to plot binned data
BINNING_DATA = 1

#: whether to to mark binned data with markers
BINNING_ERROR = 0

#: whether to plot binning error model
BINNING_MODEL = 0

#: which K value to plot
K = 16

if CROSS:

    data = np.load(os.path.join(DATA_PATH,"cross_error_corr.npy"))
    bgs = np.load(os.path.join(DATA_PATH,"cross_error_bg.npy"))
    vars = np.load(os.path.join(DATA_PATH,"cross_error_var.npy"))

    data_regular = np.load(os.path.join(DATA_PATH,"auto_standard_error_corr.npy"))[...,0:NFRAMES//PERIOD*2]
else:
    data = np.load(os.path.join(DATA_PATH,"auto_random_error_corr.npy"))
    bgs = np.load(os.path.join(DATA_PATH,"auto_random_error_bg.npy"))
    vars = np.load(os.path.join(DATA_PATH,"auto_random_error_var.npy"))

    data_regular = np.load(os.path.join(DATA_PATH,"auto_fast_error_corr.npy"))[...,0:NFRAMES]



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

#delta parameter for weight model
delta = 0.

bg1 = bg1(51,0)[...,i,j]
bg2 = bg2(51,0)[...,i,j] if CROSS else bg1
g = g1(x,51,0, cross = CROSS)[...,i,j]

wp = weight_prime_from_g(g,delta,bg1,bg2)
w = weight_from_g(g, delta)

#error estimators using a simple model of independent data (delta = 0).

err1 = sigma_prime_weighted(0., g+0j,  delta,bg1,bg2)#/n**0.5
err2 = sigma_prime_weighted(1., g,  delta,bg1,bg2)#/n**0.5
err3 = sigma_prime_weighted(wp, g, delta,bg1,bg2)#/n**0.5

err5 = sigma_weighted(0., g,  delta)#/n**0.5
err6 = sigma_weighted(1., g, delta)#/n**0.5
err7 = sigma_weighted(w, g, delta)#/n**0.5


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
    g = g1(x,51,0)[...,i,j]

    std = (((y - g)**2).mean(axis = 0))**0.5

    if binning == BINNING_DATA:
        ax1.semilogx(x[1:],y[0,1:],marker = "o", linestyle = '',fillstyle = "none",label = "$R$", color = "k")        
    if binning == BINNING_ERROR:
        ax2.semilogx(x[1:],std[1:],marker = "o", linestyle = '', fillstyle = "none",label = "$R$", color = "k")
    else:
        ax2.semilogx(x[1:],std[1:],linestyle = ':', fillstyle = "none", color = "k")




for binning in (0,1):
    ax1.set_prop_cycle(None)
    ax2.set_prop_cycle(None)
    for norm in (1,2,3,5,6,7):
    
        x,y = merge_multilevel(multilevel(data[:,norm,i,j,:],binning = binning))
        g = g1(x,51,0)[...,i,j]
        std = (((y - g)**2).mean(axis = 0))**0.5
        
        if binning == BINNING_DATA:
            ax1.semilogx(x[1:],y[0,1:],marker = MARKERS.get(norm,"o"), linestyle = '',fillstyle = "none",label = "${}$".format(LABELS.get(norm)))
        if binning == BINNING_ERROR:
            ax2.semilogx(x[1:],std[1:],marker = MARKERS.get(norm,"o"), linestyle = '', fillstyle = "none",label = "${}$".format(LABELS.get(norm)))
        else:    
            ax2.semilogx(x[1:],std[1:],linestyle = ':', fillstyle = "none")


ax1.plot(x[1:],g1(x[1:],51,0)[...,i,j], "k",label = "$g$")

# #: take first run, norm = 3 data for g estimation
# x,g = log_average(data[0,3,i,j,:])
# g = denoise(g)
# g = decreasing(g)
# g = g.clip(0,1)
# ax1.plot(x[1:],g[1:], "k:",label = "denoised")

x = np.arange(NFRAMES)
ax1.plot(x[1:],w[1:], "k--",label = "$w$")
ax1.plot(x[1:],wp[1:], "k:",label = "$w'$")

#ax2.set_ylim(ax1.get_ylim())


x,err1 = merge_multilevel(multilevel(err1,binning = 0))
x,err2 = merge_multilevel(multilevel(err2,binning = 0))
x,err3 = merge_multilevel(multilevel(err3,binning = 0))
x,err5 = merge_multilevel(multilevel(err5,binning = 0))
x,err6 = merge_multilevel(multilevel(err6,binning = 0))
x,err7 = merge_multilevel(multilevel(err7,binning = 0))

ax2.set_prop_cycle(None)


nmax = len(x)
if BINNING_MODEL or not CROSS:
    n = n[1:nmax]

ax2.loglog(x[1:],err1[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err2[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err3[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err5[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err6[1:]/np.sqrt(n),"-")
ax2.loglog(x[1:],err7[1:]/np.sqrt(n),"-")

ax2.set_xlabel(r"$\tau$")
ax2.set_ylabel(r"$\sigma$")
ax2.set_ylim(0.001,2)
ax1.set_ylabel(r"$g,w$")
ax1.set_ylim(-1,1.5)

ax1.legend(loc = 3)

plt.tight_layout()

if SAVE_FIGS:
    if CROSS:
        plt.savefig("plots/plot_cross_error_{}.pdf".format(K))
    else:
        plt.savefig("plots/plot_auto_error_{}.pdf".format(K))
        
plt.show()


