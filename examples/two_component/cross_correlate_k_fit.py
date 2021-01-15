"""Performs data fitting on cross_correlation.py data. You must
first run

$ cross_correlation.py
"""
from cddm.map import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from conf import D1, D2

SHOW_FITS = True
#: which norm modes to fit
NORMS = (2,3,6)

colors = ["C{}".format(i) for i in range(10)]

def _g1(x,f1,f2,a,b,c):
    """g1: exponential decay"""
    return a*(np.cos(b)**2 * np.exp(-f1*x) + np.sin(b)**2 * np.exp(-f2*x)) + c

def fit_data(x, data, title = "", ax = None):
    """performs fitting and plotting of cross correlation data"""
    popt = [1.2e-2,7e-5,1,0.4,0]
    for i, (k, y) in enumerate(data):
        if k > 10:
            try:
                popt,pcov = curve_fit(_g1, x, y, p0 = popt)    
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
    x = np.load("cross_correlate_t.npy")
    y = np.load("cross_correlate_norm_{}_data.npy".format(norm))
    k_data = k_select(y, angle = 0, sector = 30)
    if SHOW_FITS:
        fig = plt.figure()
        ax = fig.subplots()  
    else:
        ax = None
    results = list(fit_data(x, k_data, title = "Norm {}".format(norm),ax = ax))

    k_out = np.empty(shape = (len(results),),dtype = float)
    p_out = np.empty(shape = (len(results),5),dtype = float)
    c_out = np.empty(shape = (len(results),5,5),dtype = float)
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
    f1 = p[:,0]
    f2 = p[:,1]
    s1 = c[:,0,0]**0.5
    s2 = c[:,1,1]**0.5
    
    ax.plot((k**2),f1,"o", color = colors[i],fillstyle='none', label = "norm = {}".format(norm))
    ax.plot((k**2),f2,"o", color = colors[i],fillstyle='none')

    x = k**2
    popt,pcov = curve_fit(_lin, x, f1, sigma = s1)
    ax.plot(x,_lin(x,*popt), "--", color = colors[i], label = "fit norm = {}".format(norm))
    err = np.sqrt(np.diag(pcov))/popt
    print("Measured D2 (norm = {}): {:.3e} (1 +- {:.4f})".format(norm, popt[0], err[0]))
    x = k**2
    popt,pcov = curve_fit(_lin, x, f2, sigma = s2)
    err = np.sqrt(np.diag(pcov))/popt
    ax.plot(x,_lin(x,*popt), "--", color = colors[i])
    print("Measured D1 (norm = {}): {:.3e} (1 +- {:.4f})".format(norm, popt[0], err[0]))

ax.plot(x,_lin(x,D1), "k-", label = "true")
print("True D1: {:.3e}".format(D1))
ax.plot(x,_lin(x,D2), "k-")
print("True D2: {:.3e}".format(D2))

ax.set_xlabel("$q^2$")
ax.set_ylabel(r"$1/\tau$")
ax.legend()
plt.show()
      
