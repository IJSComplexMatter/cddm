"""
"""

from cddm.map import  k_indexmap, rfft2_grid
from cddm.fft import rfft2_crop
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from examples.paper.simple_video.conf import D, DATA_PATH, KIMAX, KJMAX
from examples.paper.simple_video.conf import *


from examples.paper.form_factor import g1

import os.path as path

SAVEFIG = True

colors = ["C{}".format(i) for i in range(10)]

colors = ["C{}".format(i) for i in range(10)]

MARKERS = {"full": "1", "standard" : "2", "fast" : "3", "dual" : "4","random" : "+"}

LABELS = {"full": r"DDM ($N={}$ @ $p = {}$)".format(NFRAMES_FULL,DT_FULL), 
          "standard": r"DDM ($N={}$ @ $p = {}$)".format(NFRAMES_STANDARD,DT_STANDARD),
          "fast": r"DDM ($N={}$ @ $p = {}$)".format(NFRAMES_FAST,DT_FAST),
          "dual": r"C-DDM ($N={}$ @ $p = {}$)".format(NFRAMES_DUAL,PERIOD),
          "random": r"R-DDM ($N={}$ @ $p = {}$)".format(NFRAMES_RANDOM,PERIOD_RANDOM)}

NORM_LABEL = {2 : "B", 3 : "C", 6 : "W"}
NORM_MARKER = {2 : ":", 3 : "--", 6 : "-"}

NORM = 6

RNORMS = (2,3,6)

DNORMS = RNORMS


#theoretical amplitude of the signal.
amp = rfft2_crop(g1(0),KIMAX,KJMAX)

def _g1(x,f,a,b):
    """g1: exponential decay"""
    return a * np.exp(-f*x) + b
          
def _fit_k(x,ys,p0):
    for y in ys:
        try:
            popt,pcov = curve_fit(_g1, x,y, p0 = p0)  
            yield popt, np.diag(pcov)
        except:
            yield (np.nan,)*3, (np.nan,)*3
    
def _fit_data(x, data, imap):
    """performs fitting and plotting of cross correlation data"""
    popt0 = [0.01,1,0]
    
    popt = np.empty((data.shape[0], data.shape[1],3),float)
    pcov = np.empty((data.shape[0], data.shape[1],3),float)
    #make all data invalid
    popt[...] = np.nan
    pcov[...] = np.nan
    
    for i in range(3, KIMAX):
        mask = imap == i
        y = data[mask,:]
        out = np.array(list(_fit_k(x,y,popt0)))
        p = out[:,0]
        c = out[:,1]
        
        popt0 = np.nanmean(p,axis = 0)
        popt[mask] = p
        pcov[mask] = c
    return popt, pcov

def _lin(x,k):
    return k*x


fig1 = plt.figure() 
ax1,ax1a = fig1.subplots(1,2)



fig2 = plt.figure() 
ax2,ax2a = fig2.subplots(1,2)


for i, method in enumerate(("standard","fast","random","dual","full")):
    
    if method == "random":
        norms = RNORMS
    elif method == "dual":
        norms = DNORMS
    else:
        norms = DNORMS#(NORM,)
        
    for norm in norms:
    
        x = np.load(path.join(DATA_PATH, "corr_{}_t.npy".format(method)))
        y = np.load(path.join(DATA_PATH, "corr_{}_data_norm{}.npy".format(method, norm)))
        
        #time mask for valid data. For a given time, all data at any k value must be valid
        mask = np.isnan(y)
        mask = np.logical_not(np.all(mask, axis = tuple(range(mask.ndim-1))))
        x,y = x[mask], y[...,mask]
        
        imap = k_indexmap(y.shape[0], y.shape[1], angle=0, sector=180, kstep=1.0)   
        ki, kj = rfft2_grid(y.shape[0], y.shape[1])
        ks = (ki**2 + kj**2)**0.5
        
     
        if method == "dual":
            popt,cov = _fit_data(x,y, imap) 
        else:
            #skip the first element (zero time)
            popt,cov = _fit_data(x[1:],y[...,1:], imap) 
            
        m = np.all( np.logical_not(np.isnan(popt)), axis = -1)
       
        popt = popt[m]
        cov = cov[m]
        k = ks[m]
        imap = imap[m]
        a_true = amp[m]
        
        #sort results based on ascending k values
        # args = np.argsort(k)
        # k = k[args]
        # popt = popt[args,:]
        # cov = cov[args,:]    
        # imap = imap[args]
        # a_true = a_true[args]
        
        f = popt[:,0]
        a = popt[:,1]
        fe= (cov[:,0])**0.5
        ae= (cov[:,1])**0.5
        
    
    
        
        
        k_err = k
        
        a_err = ae / a
        
        f_true = _lin(k**2,D)
        
        
        f_err = (f_true-f/ f_true)**2
        
        
    
        _f_err = ((f-f_true)/f_true)**2
        _a_err = ((a-a_true)/a_true)**2
        
        f_err = []
        qs = []
        a_err = []
        
        for q in range(KIMAX):
            mask = q == imap
            if np.any(mask):
                f_err.append((_f_err[mask].mean())**0.5)
                
                a_err.append(a[mask].std())
                
                #a_err.append((_a_err[mask].mean())**0.5)
                qs.append(q)
        qs = np.asarray(qs)
        f_err = np.asarray(f_err)
        a_err = np.asarray(a_err)
        m = np.asarray(f_err) < 1
            
        SKIP_SIZE = 10
        
        k2 = k**2
        
        
        if norm == NORM:
    
            ax1.plot(k2[::SKIP_SIZE],f[::SKIP_SIZE],MARKERS[method], color = colors[i],fillstyle='none', label = LABELS[method])
            ax2.plot(k[::SKIP_SIZE],a[::SKIP_SIZE],MARKERS[method], color = colors[i],fillstyle='none', label = LABELS[method])
        
            ax1a.semilogy((qs**2)[m],f_err[m],NORM_MARKER[norm]+MARKERS[method],color = colors[i],fillstyle='none', label = LABELS[method])
            ax2a.semilogy(qs[m],a_err[m],NORM_MARKER[norm]+MARKERS[method],color = colors[i],fillstyle='none', label = LABELS[method])
        
            popt,pcov = curve_fit(_lin, k2, f, sigma = fe)
            #ax1.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit {}".format(label))
            err = np.sqrt(np.diag(pcov))/popt
            print("Measured D ({}): {:.3e} (1 +- {:.4f})".format(method, popt[0], err[0]))
        else:
            ax1a.semilogy((qs**2)[m],f_err[m],NORM_MARKER[norm],color = colors[i],fillstyle='none')
            ax2a.semilogy(qs[m],a_err[m],NORM_MARKER[norm],color = colors[i],fillstyle='none')
                   

x = np.arange(KIMAX)**2
ax1.plot(x,_lin(x,D), "k-")


def ampmean(amp):
    imap = k_indexmap(amp.shape[0], amp.shape[1], angle=0, sector=180, kstep=1.0)
    for i in range(KIMAX):
        mask = imap == i
        yield amp[mask].mean()

y = list(ampmean(amp))

ax2.plot(y, "k-")
print("True D: {:.3e}".format(D))

ax1.set_xlabel("$q^2$")
ax1.set_ylabel(r"$1/\tau_0$")
ax1.set_ylim(0,0.25)

ax2.set_xlabel("$q$")

ax1a.set_ylabel("err")
ax2a.set_ylabel("err")

# ax2.set_ylim(0.01,10)
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title(r"$1/\tau_0(q)$")
ax1a.set_title(r"err$(q)$")

# ax1a.set_xlabel("$q$")
ax2.set_ylabel(r"$a$")
ax2.set_ylim(0.,1.1)
# ax2a.set_ylabel("$\sigma$")
ax2a.set_xlabel("$q$")
ax1a.set_xlabel("$q^2$")
# ax2a.set_ylim(0.001,100)
ax1a.legend(prop={'size': 8})
ax2a.legend(prop={'size': 8})
ax2.set_title(r"$a(q)$")
ax2a.set_title(r"err$(q)$")
# ax2a.set_title(r"$\sigma(q)$")


fig1.tight_layout()
fig2.tight_layout()

if SAVEFIG:
    fig1.savefig("plots/plot_fit_rate.pdf")
    fig2.savefig("plots/plot_fit_amplitude.pdf")

plt.show()
      

