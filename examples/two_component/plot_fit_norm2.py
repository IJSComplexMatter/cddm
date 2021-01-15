"""
"""

from cddm.map import  k_indexmap, rfft2_grid
from cddm.fft import rfft2_crop
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from examples.paper.two_component.conf import D2, DATA_PATH, KIMAX, KJMAX
from examples.paper.two_component.conf import *


from  conf import *

from examples.paper.form_factor import g1

import os.path as path

SAVEFIG = True

colors = ["C{}".format(i) for i in range(10)]

LABELS = {0: "B'", 1 : "C'", 2 : "B", 3 : "C",4 : "W'", 6 : "W"}
MARKERS = {0: "1", 1 : "2", 2 : "3", 3 : "4",4 : "+", 6 : "x"}


METHOD = "dual"


amp = rfft2_crop(g1(0),KIMAX,KJMAX)

def _g1(x,f1,a,f2,b):
    """g1: exponential decay"""
    return a * (b*np.exp(-f1*x) + (1-b) * np.exp(-f2*x))

            
def _fit_k(x,ys,p0):
    for y in ys:
        try:
            popt,pcov = curve_fit(_g1, x,y, p0 = p0)  
            yield popt, np.diag(pcov)
        except:
            yield (np.nan,)*4, (np.nan,)*4
    
def _fit_data(x, data, imap):
    """performs fitting and plotting of cross correlation data"""
    popt0 = [0.0001,1,0.1,0.5]
    
    popt = np.empty((data.shape[0], data.shape[1],4),float)
    pcov = np.empty((data.shape[0], data.shape[1],4),float)
    #make all data invalid
    popt[...] = np.nan
    pcov[...] = np.nan
    
    for i in range(6, KIMAX):
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
for i, norm in enumerate((2,3,6)):
    
    
    #x = np.load(path.join(DATA_PATH, "corr_{}_t.npy".format(METHOD)))
    #y = np.load(path.join(DATA_PATH, "corr_{}_data_norm{}.npy".format(METHOD, norm)))

    x = np.load( "cross_correlate_t.npy")
    y = np.load("cross_correlate_norm_{}_data.npy".format(norm))

    
    #time mask for valid data. For a given time, all data at any k value must be valid
    mask = np.isnan(y)
    mask = np.logical_not(np.all(mask, axis = tuple(range(mask.ndim-1))))
    x,y = x[mask], y[...,mask]
    
    if METHOD == "dual":
        m,ks,p,c = fit(x,y, label = norm) 
    else:
        #skip the first element (zero time)
        m,ks,p,c = fit(x[1:],y[...,1:], label = norm) 
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
    
    f_true = _lin(k**2,D1)
    
    KERNEL_SIZE = 40
    kernel = (1/KERNEL_SIZE,)*KERNEL_SIZE
    
    f_err = (np.convolve(((f-f_true)/f_true)**2, kernel ,mode = "valid") ** 0.5) 
    k_err = np.convolve(k, kernel ,mode = "valid")
    
    a_err = (np.convolve(((a-a_true)/a_true)**2, kernel ,mode = "valid") ** 0.5) 
    
    
    ax1.plot((k**2)[::KERNEL_SIZE],f[::KERNEL_SIZE],MARKERS[norm], color = colors[i],fillstyle='none', label = LABELS[norm])

    ax2.plot(k[::KERNEL_SIZE],a[::KERNEL_SIZE],MARKERS[norm], color = colors[i],fillstyle='none', label = LABELS[norm])



    ax1a.semilogy((k_err**2)[::KERNEL_SIZE],f_err[::KERNEL_SIZE],"-",color = colors[i],fillstyle='none', label = LABELS[norm])
    ax2a.semilogy(k_err[::KERNEL_SIZE],a_err[::KERNEL_SIZE],"-",color = colors[i],fillstyle='none', label = LABELS[norm])

    
    
    
    popt,pcov = curve_fit(_lin, k**2, f,sigma = fe)
    ax1.plot(k**2,_lin(k**2,*popt), "--", color = colors[i])
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D ({}): {:.3e} (1 +- {:.4f})".format(norm, popt[0], err[0]))

ax1.plot(k**2,_lin(k**2,D1), "k--", label = "expected value")



def ampmean(amp):
    imap = k_indexmap(amp.shape[0], amp.shape[1], angle=0, sector=180, kstep=1.0)
    for i in range(KIMAX):
        mask = imap == i
        yield amp[mask].mean()


y = list(ampmean(amp))

ax2.plot(y, "k--", label = "expected value")
print("True D: {:.3e}".format(D1))

ax1.set_xlabel("$q^2$")
ax1.set_ylabel(r"$1/\tau_0$")
ax1.set_ylim(0,2)

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
    fig1.savefig("fit_rate_{}.pdf".format(METHOD))
    fig2.savefig("fit_amplitude_{}.pdf".format(METHOD))

plt.show()
      

