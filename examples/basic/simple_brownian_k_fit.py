"""Performs data fitting.

To generate test data, you must first run the following scripts

simple_brownian_ccorr.py
simple_brownian_ccorr_multi.py
simple_brownian_acorr.py
"""

from cddm.map import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from simple_brownian_video import D

colors = ["C{}".format(i) for i in range(10)]


#c = np.load("simple_brownian_ccorr_linear.npy")
#xc = np.arange(c.shape[-1])
#
xc0 = np.load("simple_brownian_ccorr_norm0_log_x.npy")
c0 = np.load("simple_brownian_ccorr_norm0_log_data.npy")
#
xc1 = np.load("simple_brownian_ccorr_norm1_log_x.npy")
c1 = np.load("simple_brownian_ccorr_norm1_log_data.npy")
a = np.load("simple_brownian_acorr_log_data.npy")
xa = np.load("simple_brownian_acorr_log_x.npy")

##now do some k-averaging over a cone of 5 degrees, at 0 angle
##kc and ka are lists of (q,data) tuples

kc0 = list(k_select(c0, angle = 0, sector = 5, kstep = 1))
kc1 = list(k_select(c1, angle = 0, sector = 5, kstep = 1))
ka = list(k_select(a, angle = 0, sector = 5, kstep = 1))


#Plot 16th and 28th q
plt.figure()

for i in (15,27):
    k, y = ka[i]
    plt.semilogx(xa[1:], y[1:]/y[0], label = "acorr, k = {:.1f}".format(k))    

for i in (15,27):
    k, y = kc0[i]
    plt.semilogx(xc0[1:], y[1:]/y[0], label = "ccorr, norm = 0, k = {:.1f}".format(k))    

for i in (15,27):
    k, y = kc1[i]
    plt.semilogx(xc1[1:], y[1:]/y[0], label = "ccorr, norm = 1, k = {:.1f}".format(k))    


plt.legend()


def _fit(x,f,a,b):
    return a*np.exp(-f*x) + b

def fitc(x, data, title = ""):
    """performs fitting and plotting of cross correlation data"""
    popt = [0.1,1,0]
    for i, (k, y) in enumerate(data):
        y = y/y[0]
        try:
            popt,pcov = curve_fit(_fit, x, y, p0 = popt)
            
            plt.semilogx(x,y,"o",color = colors[i%len(colors)],fillstyle='none')
            plt.semilogx(x,_fit(x,*popt), color = colors[i%len(colors)])
            yield k, popt[0]
        except:
            pass
    plt.title(title)
        
def fita(x, data, title = ""):
    """performs fitting and plotting of auto correlation data"""
    popt = [0.1,1,0]
    for i, (k, y) in enumerate(data):
        y = y/y[0]
        try:
            popt,pcov = curve_fit(_fit, x[1:], y[1:], p0 = popt)
            
            plt.semilogx(x,y,"o", color = colors[i%len(colors)],fillstyle='none')
            plt.semilogx(x,_fit(x,*popt), color = colors[i%len(colors)])
            yield k, popt[0]
        except:
            pass
    plt.title(title)
  
def _lin(x,k):
    return k*x
      
plt.figure()  

kfc0 = np.array(list(fitc(xc0,kc0[4:],"ccorr, norm = 0")))
plt.figure() 
kfa = np.array(list(fita(xa,ka[4:],"acorr")))
plt.figure() 
kfc1 = np.array(list(fitc(xc1,kc1[4:],"ccorr, norm = 1")))
plt.figure() 


plt.plot(kfa[...,0]**2,kfa[...,1],"o", color = colors[0],fillstyle='none', label = "acorr")
plt.plot(kfc0[...,0]**2,kfc0[...,1],"o", color = colors[1],fillstyle='none', label = "ccorr0")
plt.plot(kfc1[...,0]**2,kfc1[...,1],"o", color = colors[2],fillstyle='none', label = "ccorr1 ")

x = kfc0[...,0]**2


plt.plot(x,_lin(x,D), "k-", label = "true")

popt,pcov = curve_fit(_lin, x, kfc0[...,1])
plt.plot(x,_lin(x,*popt), "--", color = colors[1], label = "fit ccorr0")
print("Measured D (ccorr, norm = 0):", popt[0])

popt,pcov = curve_fit(_lin, x, kfc1[...,1])
plt.plot(x,_lin(x,*popt), "--", color = colors[2], label = "fit ccorr1")
print("Measured D (ccorr, norm = 1):", popt[0])

popt,pcov = curve_fit(_lin, x, kfa[...,1])
plt.plot(x,_lin(x,*popt), "--",color = colors[0], label = "fit acorr")
print("Measured D (acorr):", popt[0])

print("Expected D:", D)

plt.xlabel("$q^2$")

plt.ylabel(r"$1/\tau$")
plt.legend()
plt.show()


        
        
    



