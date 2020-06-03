"""
"""

from cddm.map import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from examples.paper.conf import D, DATA_PATH
import os.path as path

SHOW_FITS = True

colors = ["C{}".format(i) for i in range(10)]

def _g1(x,f,a,b):
    """g1: exponential decay"""
    return a* np.exp(-f*x) + b

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


def fit(x,y, title = "data"):
    k_data = k_select(y, angle = 0, sector = 180, kstep = 1)
    if SHOW_FITS:
        fig = plt.figure()
        ax = fig.subplots()  
    else:
        ax = None
    results = list(fit_data(x, k_data, title = title ,ax = ax))

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

for i,label in enumerate(("standard", "random", "dual")):
    x = np.load(path.join(DATA_PATH, "corr_{}_t.npy".format(label)))
    y = np.load(path.join(DATA_PATH, "corr_{}_data.npy".format(label)))
    
    mask = np.isnan(y)
    mask = np.logical_not(np.all(mask, axis = tuple(range(mask.ndim-1))))
    x,y = x[mask], y[...,mask]

    k,p,c = fit(x[1:],y[...,1:], title = label) 
    f = p[:,0]
    
    ax.plot((k**2),f,"o", color = colors[i],fillstyle='none', label = "{}".format(label))

    x = k**2
    popt,pcov = curve_fit(_lin, x, f)
    ax.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit {}".format(label))
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D (norm = {}): {:.3e} (1 +- {:.4f})".format(label, popt[0], err[0]))

ax.plot(x,_lin(x,D), "k-", label = "true")
print("True D: {:.3e}".format(D))

ax.set_xlabel("$q^2$")
ax.set_ylabel(r"$1/\tau$")
ax.legend()
plt.show()
      

