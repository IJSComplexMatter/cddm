"""
"""

from cddm.map import  k_indexmap, rfft2_grid
from cddm.fft import rfft2_crop
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from examples.paper.one_component.conf import D, DATA_PATH, KIMAX, KJMAX
from examples.paper.one_component.conf import *


from examples.paper.form_factor import g1

import os.path as path

SAVEFIG = True

colors = ["C{}".format(i) for i in range(10)]

MARKERS = {"full": "1", "standard" : "2", "fast" : "3", "dual" : "4","random" : "+"}

LABELS = {"full": r"DDM ($N={}$ @ $\delta t = {}$)".format(NFRAMES_FULL,DT_FULL), 
          "standard": r"DDM ($N={}$ @ $\delta t = {}$)".format(NFRAMES_STANDARD,DT_STANDARD),
          "fast": r"DDM ($N={}$ @ $\delta t = {}$)".format(NFRAMES_FAST,DT_FAST),
          "dual": r"C-DDM ($N={}$ @ $p = {}$)".format(NFRAMES_DUAL,PERIOD),
          "random": r"R-DDM ($N={}$ @ $p = {}$)".format(NFRAMES_RANDOM,PERIOD_RANDOM)}

NORM = 6

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


def fit(x,y, label = "data"):

    imap = k_indexmap(y.shape[0], y.shape[1], angle=0, sector=180, kstep=1.0)        
    popt, pcov = _fit_data(x, y, imap)
    ki, kj = rfft2_grid(y.shape[0], y.shape[1])
    k = (ki**2 + kj**2)**0.5
    
    #mask of valid (successfully fitted) data
    mask =np.all( np.logical_not(np.isnan(popt)), axis = -1)
    
    return mask, k, popt, pcov

fig1 = plt.figure() 
ax1,ax1a = fig1.subplots(1,2)

#ax1 =  ax1a.twinx()

fig2 = plt.figure() 
ax2,ax2a = fig2.subplots(1,2)

#ax2 =  ax2a.twinx()


#for i,label in enumerate(("standard", "random", "dual")):
for i,label in enumerate(("full","standard","fast","dual","random")):
    x = np.load(path.join(DATA_PATH, "corr_{}_t.npy".format(label)))
    y = np.load(path.join(DATA_PATH, "corr_{}_data_norm{}.npy".format(label, NORM)))
    
    #time mask for valid data. For a given time, all data at any k value must be valid
    mask = np.isnan(y)
    mask = np.logical_not(np.all(mask, axis = tuple(range(mask.ndim-1))))
    x,y = x[mask], y[...,mask]
    
    if label == "dual":
        m,ks,p,c = fit(x,y, label = label) 
    else:
        #skip the first element (zero time)
        m,ks,p,c = fit(x[1:],y[...,1:], label = label) 
    f = p[m,0]
    a = p[m,1]
    k = ks[m]
    
    a_true = amp[m]
    
    fe= (c[m,0])**0.5
    ae= (c[m,1])**0.5
    
    from operator import itemgetter
    
    s = np.array(sorted(((k,f,fe,a,ae,at) for (k,f,fe,a,ae,at) in zip(k,f,fe,a,ae,a_true)),key=itemgetter(0)))

    k = s[:,0]
    f = s[:,1]
    fe = s[:,2]
    a = s[:,3]
    ae = s[:,4]
    a_true = s[:,5]
    
    
    f_err = fe / f
    k_err = k
    
    a_err = ae / a
    
    f_true = _lin(k**2,D)
    
    KERNEL_SIZE = 30
    kernel = (1/KERNEL_SIZE,)*KERNEL_SIZE
    
    f_err = (np.convolve(((f-f_true)/f_true)**2, kernel ,mode = "valid") ** 0.5) 
    k_err = np.convolve(k, kernel ,mode = "valid")
    
    a_err = (np.convolve(((a-a_true)/a_true)**2, kernel ,mode = "valid") ** 0.5) 
    
    
    ax1.plot((k**2)[::KERNEL_SIZE],f[::KERNEL_SIZE],MARKERS[label], color = colors[i],fillstyle='none', label = LABELS[label])

    ax2.plot(k[::KERNEL_SIZE],a[::KERNEL_SIZE],MARKERS[label], color = colors[i],fillstyle='none', label = LABELS[label])



    ax1a.semilogy((k_err**2)[::KERNEL_SIZE],f_err[::KERNEL_SIZE],"-",color = colors[i],fillstyle='none', label = LABELS[label])
    ax2a.semilogy(k_err[::KERNEL_SIZE],a_err[::KERNEL_SIZE],"-",color = colors[i],fillstyle='none', label = LABELS[label])

    
    
    x = k**2
    
    
    popt,pcov = curve_fit(_lin, x, f, sigma = fe)
    #ax1.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit {}".format(label))
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D ({}): {:.3e} (1 +- {:.4f})".format(label, popt[0], err[0]))

ax1.plot(x,_lin(x,D), "k--", label = "expected value")



def ampmean(amp):
    imap = k_indexmap(amp.shape[0], amp.shape[1], angle=0, sector=180, kstep=1.0)
    for i in range(KIMAX):
        mask = imap == i
        yield amp[mask].mean()


y = list(ampmean(amp))

ax2.plot(y, "k--", label = "expected value")
print("True D: {:.3e}".format(D))

ax1.set_xlabel("$q^2$")
ax1.set_ylabel(r"$1/\tau_0$")
ax1.set_ylim(0,0.15)

ax1a.set_ylabel("$err$")
ax2a.set_ylabel("$err$")

# ax2.set_ylim(0.01,10)
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title(r"$1/\tau_0(q)$")
ax1a.set_title(r"err$(q)$")

# ax1a.set_xlabel("$q$")
ax2.set_ylabel(r"$a$")
ax2.set_ylim(0.6,1.1)
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
    fig1.savefig("fit_rate_norm{}.pdf".format(NORM))
    fig2.savefig("fit_amplitude_norm{}.pdf".format(NORM))

plt.show()
      

