"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform cross-ddm analysis.

You must first create FFTs of the videos by calling simple_brownian_fft.py
"""

from simple_brownian_video import PERIOD

from cddm import normalize, k_select, cdiff_multi, log_merge, ccorr_multi

import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#setting this to 2 shows progress bar
conf.set_verbose(2)

SHAPE = (512, 512)

v1 = np.load("simple_brownian_cddm_fft_0.npy")
v2 = np.load("simple_brownian_cddm_fft_1.npy")
v1 = v1/(v1[...,0,0][:,None,None])
v2 = v2/(v2[...,0,0][:,None,None])

v1 = v1/v1[...,0,0].mean()
v2 = v2/v2[...,0,0].mean()

v1 = v1 - v1.mean(axis = 0)[None,...]
v2 = v2 - v2.mean(axis = 0)[None,...]

t1 = np.load("simple_brownian_cddm_t1.npy")
t2 = np.load("simple_brownian_cddm_t2.npy")

nframes = len(v1)

data = cdiff_multi(v1,v2 , t1,t2, n = 16, period = PERIOD, binning = True)

cfast, cslow = normalize(data)


i,j = 4,4

plt.figure()


x = np.arange(cfast.shape[-1])

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)

plt.semilogx(x[1:], cfast[i,j][1:], "o", label = "fast", fillstyle = "none")

#plot slow data
x = np.arange(cslow.shape[-1]) * PERIOD
for n, slow in enumerate(cslow):
    x = x * 2
    plt.semilogx(x[1:], slow[i,j][1:], "o", label = "slow {}".format(n+1), fillstyle = "none")
    
#merged data
x, logdata = log_merge(cfast,cslow)
plt.semilogx(x[1:], logdata[i,j][1:], "k-", label = "merged")

plt.legend()

np.save("simple_brownian_cdiff_log.npy",(x,logdata))

##now let us do some k-averaging
kdata = k_select(logdata, phi = 15, sector = 3, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    plt.semilogx(x[1:], c[1:]/c[0], label = k)
plt.legend()
plt.show()



