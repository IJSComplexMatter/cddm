"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform cross-ddm analysis.

You must first create FFTs of the videos by calling simple_brownian_fft.py
"""

from simple_brownian_video import PERIOD

from cddm import normalize, k_select, ccorr_multi, log_merge, iccorr_multi
from cddm.video import fromarrays
import matplotlib.pyplot as plt

from cddm.core import normalize_ccorr, cdiff_multi

from cddm import conf
import numpy as np

np.random.seed(1)
#setting this to 2 shows progress bar
conf.set_verbose(2)

SHAPE = (512, 512)

v1 = np.load("simple_brownian_cddm_fft_0.npy")
v2 = np.load("simple_brownian_cddm_fft_1.npy")
#v1 = v1/(v1[...,0,0][:,None,None])
#v2 = v2/(v2[...,0,0][:,None,None])
#

v1 = v1/v1[...,0,0].mean()
v2 = v2/v2[...,0,0].mean()

#v1[3000:] = v1[3000:] + v1[0]
#v2[3000:] = v2[3000:] + v2[-1]

v1 = v1+ np.random.randn(64,33)
v2 = v2+ np.random.randn(64,33)

v1 = v1 - v1[:2].mean(axis = 0)[None,...]
v2 = v2 - v2[:2].mean(axis = 0)[None,...]

t1 = np.load("simple_brownian_cddm_t1.npy")
t2 = np.load("simple_brownian_cddm_t2.npy")

nframes = len(v1)

v = fromarrays((v1,v2))

data, bg, var = iccorr_multi(v, t1,t2, level = 5, chunk_size = 64, auto_background = True, period = PERIOD, binning = True, stats = True, norm = 1, show = True)
#data, bg, var = ccorr_multi(v1,v2 , t1,t2, n=2**5, period = PERIOD, binning = True, norm = 0, stats = True)
#data2 = ccorr_multi(v1,v2 , t1,t2, n=2**4, period = PERIOD, binning = False, norm = 2)

v = fromarrays((v1,v2))
data2, bg2, var2 = ccorr_multi(v1,v2 , t1,t2, n =2**4, period = PERIOD, binning = True, norm = 2, stats = True)

cfast2, cslow2 = normalize_ccorr(data2,bg2,var2)
cfast, cslow = normalize_ccorr(data,bg,var)



i,j = 0,6

plt.figure()


x = np.arange(cfast.shape[-1])

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)

plt.semilogx(x[1:], cfast[i,j][1:], "o", label = "fast - level 0", fillstyle = "none")

#plot slow data
x = np.arange(cslow.shape[-1]) * PERIOD
for n, slow in enumerate(cslow):
    x = x * 2
    plt.semilogx(x, slow[i,j], "o", label = "slow - level {}".format(n+1), fillstyle = "none")
    
#merged data
x, logdata = log_merge(cfast,cslow)

#np.save("simple_brownian_ccorr_log.npy",(x,logdata))

plt.semilogx(x[1:], logdata[i,j][1:], "k-", label = "merged")


x2, logdata2 = log_merge(cfast2,cslow2)

plt.semilogx(x2[1:], logdata2[i,j][1:], "k--", label = "merged2")
plt.legend()

#np.save("simple_brownian_ccorr_log2.npy",(x2,logdata2))

##now let us do some k-averaging
kdata = k_select(logdata, phi = 90, sector = 120, kstep = 1)
kdata2 = k_select(logdata2, phi = 90, sector = 120, kstep = 1)

plt.figure()
#
for i,(k, c) in enumerate(kdata): 
    print(k)
    if k >29:

        plt.semilogx(x[1:], c[1:], label = k)
    
for i,(k, c) in enumerate(kdata2): 
    print(k)
    if k >29:   
        plt.semilogx(x2[1:], c[1:], label = k)    
    
plt.legend()
plt.show()



