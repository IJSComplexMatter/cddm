"""Performs data fitting.

To generate test data, you must first run the following scripts

simple_brownian_ccorr.py
simple_brownian_ccorr_multi.py
simple_brownian_acorr.py
"""

from cddm import k_select
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


colors = ["C{}".format(i) for i in range(10)]


c = np.load("simple_brownian_ccorr_linear.npy")
xc = np.arange(c.shape[-1])

xcl,cl = np.load("simple_brownian_ccorr_log.npy", allow_pickle = True)
xcl2,cl2 = np.load("simple_brownian_ccorr_log2.npy", allow_pickle = True)

a = np.load("simple_brownian_acorr_linear.npy")
xa = np.arange(a.shape[-1])

##now do some k-averaging over a cone of 5 degrees, at 0 angle
##kc and ka are lists of (q,data) tuples
kc = list(k_select(c, phi = 0, sector = 0, kstep = 1))
kcl = list(k_select(cl, phi = 0, sector = 0, kstep = 1))
kcl2 = list(k_select(cl2, phi = 0, sector = 0, kstep = 1))
ka = list(k_select(a, phi = 0, sector = 0, kstep = 1))


#Plot 16th and 28th q
plt.figure()
for i in (15,27):
    k, y = kc[i]
    plt.semilogx(xc[1:], y[1:]/y[0], label = "ccorr k = {:.1f}".format(k))

for i in (15,27):
    k, y = ka[i]
    plt.semilogx(xa[1:], y[1:]/y[0], label = "acorr k = {:.1f}".format(k))    

for i in (15,27):
    k, y = kcl[i]
    plt.semilogx(xcl[1:], y[1:]/y[0], label = "ccorr_multi k = {:.1f}".format(k))    

for i in (15,27):
    k, y = kcl2[i]
    plt.semilogx(xcl2[1:], y[1:]/y[0], label = "ccorr_multi2 k = {:.1f}".format(k))    


plt.legend()


def _fit(x,f,a,b):
    return a*np.exp(-f*x) + b

def fitc(x, data):
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
        
def fita(x, data):
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
  
def _lin(x,k):
    return k*x
      
plt.figure()  

kfc = np.array(list(fitc(xc,kc[4:])))
plt.figure() 
kfa = np.array(list(fita(xa,ka[4:])))
plt.figure() 
kfcl = np.array(list(fitc(xcl,kcl[4:])))
plt.figure() 
kfcl2 = np.array(list(fitc(xcl2,kcl2[4:])))
plt.figure() 

plt.plot(kfa[...,0]**2,kfa[...,1],"o", color = colors[0],fillstyle='none', label = "acorr")
plt.plot(kfc[...,0]**2,kfc[...,1],"o", color = colors[1],fillstyle='none', label = "ccorr")
plt.plot(kfcl[...,0]**2,kfcl[...,1],"o", color = colors[2],fillstyle='none', label = "ccorr_multi")
plt.plot(kfcl2[...,0]**2,kfcl2[...,1],"o", color = colors[3],fillstyle='none', label = "ccorr_multi2")

x = kfc[...,0]**2

D = 2*(4*np.pi/512)**2
plt.plot(x,_lin(x,D), "k-", label = "true")

popt,pcov = curve_fit(_lin, x, kfc[...,1])
plt.plot(x,_lin(x,*popt), "--", color = colors[1], label = "fit ccorr")
print("Measured D (ccorr):", popt[0])

popt,pcov = curve_fit(_lin, x, kfcl[...,1])
plt.plot(x,_lin(x,*popt), "--", color = colors[2], label = "fit ccorr_multi")
print("Measured D (ccorr_multi):", popt[0])

popt,pcov = curve_fit(_lin, x, kfcl2[...,1])
plt.plot(x,_lin(x,*popt), "--", color = colors[3], label = "fit ccorr_multi2")
print("Measured D (ccorr_multi2):", popt[0])

popt,pcov = curve_fit(_lin, x, kfa[...,1])
plt.plot(x,_lin(x,*popt), "--",color = colors[0], label = "fit acorr")
print("Measured D (acorr):", popt[0])

print("Expected D:", D)

plt.xlabel("$q^2$")

plt.ylabel(r"$1/\tau$")
plt.legend()
plt.show()


        
        
    



