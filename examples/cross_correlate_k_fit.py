"""Performs data fitting on cross_correlation_multi_live.py data. You must
first run

$ cross_correlation_multi_live.py
"""

from cddm.map import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from examples.conf import D, DATA_PATH
import os.path as path

SHOW_FITS = False
#: which norm modes to fit
NORMS = (2,5,6,7)

colors = ["C{}".format(i) for i in range(10)]

def _g1(x,f,a,b):
    """g1: exponential decay"""
    return a* np.exp(-f*x)  + b

def fit_data(x, data, title = "", ax = None):
    """performs fitting and plotting of cross correlation data"""
    popt = [0.1,1,0]
    for i, (k, y) in enumerate(data):
        try:
            popt,pcov = curve_fit(_g1, x,y, p0 = popt)    
            if ax is not None:
                ax.semilogx(x,y,"o",color = colors[i%len(colors)],fillstyle='none')
                ax.semilogx(x,_g1(x,*popt), color = colors[i%len(colors)])
            yield k, popt, pcov
        except:
            pass
    if ax is not None:
        ax.set_title(title)
    
def _lin(x,k):
    return k*x

def load_and_fit(norm):
    x = np.load(path.join(DATA_PATH,"cross_correlate_multi_t.npy"))
    y = np.load(path.join(DATA_PATH,"cross_correlate_multi_norm_{}_data.npy".format(norm)))
    k_data = k_select(y, angle = 0, sector = 15)
    if SHOW_FITS:
        fig = plt.figure()
        ax = fig.subplots()  
    else:
        ax = None
    results = list(fit_data(x, k_data, title = "Norm {}".format(norm),ax = ax))

    k_out = np.empty(shape = (len(results),),dtype = float)
    p_out = np.empty(shape = (len(results),3),dtype = float)
    c_out = np.empty(shape = (len(results),3,3),dtype = float)
    results = np.array(list(fit_data(x, k_data)))
    for i,(k,p,c) in enumerate(results):
        k_out[i] = k
        p_out[i,:] = p
        c_out[i,:,:] = c
        
    return k_out, p_out, c_out

fig = plt.figure() 
ax = fig.subplots()

for i,norm in enumerate(NORMS):

    k,p,c = load_and_fit(norm) 
    f = p[:,0]
    
    ax.plot((k**2),f,"o", color = colors[i],fillstyle='none', label = "norm = {}".format(norm))

    x = k**2
    popt,pcov = curve_fit(_lin, x, f)
    ax.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit norm = {}".format(norm))
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D (norm = {}): {:.3e} (1 +- {:.4f})".format(norm, popt[0], err[0]))

ax.plot(x,_lin(x,D), "k-", label = "true")
print("True D: {:.3e}".format(D))

ax.set_xlabel("$q^2$")
ax.set_ylabel(r"$1/\tau$")
ax.legend()
plt.show()
      

