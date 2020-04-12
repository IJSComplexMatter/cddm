"""Performs data fitting on cross_correlation_multi_live.py data. You must
first run

$ cross_correlation_multi_live.py
"""

from cddm.map import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

#diffusion constant
from conf import D

SHOW_FITS = False

colors = ["C{}".format(i) for i in range(10)]

def _g1(x,f,a,b):
    """g1: exponential decay"""
    return a*np.exp(-f*x) + b

def fit_data(x, data, title = ""):
    """performs fitting and plotting of cross correlation data"""
    popt = [0.1,1,0]
    if SHOW_FITS:
        plt.figure()
    for i, (k, y) in enumerate(data):
        try:
            popt,pcov = curve_fit(_g1, x, y, p0 = popt)    
            if SHOW_FITS == True:
                plt.semilogx(x,y,"o",color = colors[i%len(colors)],fillstyle='none')
                plt.semilogx(x,_g1(x,*popt), color = colors[i%len(colors)])
            yield k, popt[0]
        except:
            pass
    if SHOW_FITS:
        plt.title(title)
    
def _lin(x,k):
    return k*x

def load_and_fit(norm):
    x = np.load("cross_correlate_multi_norm_{}_t.npy".format(norm))
    y = np.load("cross_correlate_multi_norm_{}_data.npy".format(norm))
    k_data = k_select(y, angle = 0, sector = 180)
    results = np.array(list(fit_data(x, k_data)))
    return results[:,0], results[:,1]


k_1, rate_1 = load_and_fit(1) 
k_2, rate_2 = load_and_fit(2) 
k_3, rate_3 = load_and_fit(3) 

plt.figure()  

#k_1, rate_1 = k_1[:-2], rate_1[:-2] #remove high wavenumber data

plt.plot((k_1**2),rate_1,"o", color = colors[0],fillstyle='none', label = "norm = 1")
plt.plot(k_2**2,rate_2,"o", color = colors[1],fillstyle='none', label = "norm = 2")
plt.plot(k_3**2,rate_3,"o", color = colors[2],fillstyle='none', label = "norm = 3")

x = k_1**2
popt,pcov = curve_fit(_lin, x, rate_1)
plt.plot(x,_lin(x,*popt), "--", color = colors[0], label = "fit norm = 1")
print("Measured D (ccorr, norm = 1):", popt[0])

x = k_2**2
popt,pcov = curve_fit(_lin, x, rate_2)
plt.plot(x,_lin(x,*popt), "--", color = colors[1], label = "fit norm = 2")
print("Measured D (ccorr, norm = 2):", popt[0])

x = k_3**2
popt,pcov = curve_fit(_lin, x, rate_3)
plt.plot(x,_lin(x,*popt), "--", color = colors[2], label = "fit norm = 3")
print("Measured D (ccorr, norm = 3):", popt[0])

plt.plot(x,_lin(x,D), "k-", label = "true")
print("True D:", D)

plt.legend()

plt.xlabel("$q^2$")

plt.ylabel(r"$1/\tau$")
plt.legend()
plt.show()
      
