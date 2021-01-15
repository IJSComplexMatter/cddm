"""
"""

from cddm.map import  k_indexmap, rfft2_grid
from cddm.fft import rfft2_crop
from cddm.multitau import log_average
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

LABELS = {0: "B'", 1 : "C'", 2 : "B", 3 : "C",4 : "W'", 6 : "W"}
MARKERS = {0: "1", 1 : "2", 2 : "3", 3 : "4",4 : "+", 6 : "x"}


METHOD = "random"


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


y = np.load("cross_error_corr.npy")
vars = np.load("cross_error_var.npy")
var = vars.mean(axis = 1, keepdims = True)
y0 = y#/var[...,None]
x0 = np.arange(y.shape[-1])

#x0, y0 = log_average(y0)

mout = np.empty(y0.shape[:-1])
pout = np.empty(y0.shape[:-1]+ (3,))
cout = np.empty(y0.shape[:-1]+ (3,))


print("fitting...")
#for i,label in enumerate(("standard", "random", "dual")):
for i in range(y.shape[0]):
    for j, norm in enumerate((2,3,6)):
    
        #time mask for valid data. For a given time, all data at any k value must be valid
        mask = np.isnan(y0[i,norm])
        mask = np.logical_not(np.all(mask, axis = tuple(range(mask.ndim-1))))
        x,y = x0[mask], y0[i,norm][...,mask]
        
        x, y = x0, y0[i,norm]

        if METHOD == "dual":
            m,ks,p,c = fit(x,y, label = norm) 
        else:
            #skip the first element (zero time)
            m,ks,p,c = fit(x[1:],y[...,1:], label = norm) 
            
        mout[i,j] = m
        pout[i,j] = p
        cout[i,j] = c


for i, norm in enumerate((2,3,6)):  
    

    popt = pout[:,i,]
    
    m = np.logical_not(np.all(np.isnan(popt), axis = -1))
    m = np.any(m, axis = 0)
    
    a_true = amp[:,0:1][m]
    
    k = ks[m]

    popt = pout[:,i,m]
    
    
    f = popt[...,0]
    a = popt[...,1]
    
    f_true = _lin(k**2,D)

    f_err = ((f-f_true)/f_true)**2
    f_err = np.nanmean(f_err, axis = 0) ** 0.5
    #f = np.nanmean(f,axis = 0)


    a_err = ((a-a_true)/a_true)**2
    a_err = np.nanmean(a_err, axis = 0) ** 0.5

    ax1.plot((k**2),f.mean(axis = 0),MARKERS[norm], color = colors[i],fillstyle='none')

    ax2.plot(k,a.mean(axis= 0),MARKERS[norm], color = colors[i],fillstyle='none')



    ax1a.semilogy((k**2),f_err,"-",color = colors[i],fillstyle='none', label = LABELS[norm])
    ax2a.semilogy(k,a_err,"-",color = colors[i],fillstyle='none', label = LABELS[norm])


    
    x = k**2
    
    
    popt,pcov = curve_fit(_lin, x, f.mean(axis = 0))
    #ax1.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit {}".format(label))
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D ({}): {:.3e} (1 +- {:.4f})".format(norm, popt[0], err[0]))

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
ax1.set_ylim(0,0.2)

ax1a.set_ylabel("$err$")
ax2a.set_ylabel("$err$")

# ax2.set_ylim(0.01,10)
ax1.legend(prop={'size': 8})
ax2.legend(prop={'size': 8})
ax1.set_title(r"$1/\tau_0(q)$")
ax1a.set_title(r"err$(q)$")

# ax1a.set_xlabel("$q$")
ax2.set_ylabel(r"$a$")

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
      

