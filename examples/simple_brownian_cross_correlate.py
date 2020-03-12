"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform cross-ddm analysis.

You must first create FFTs of the videos by calling simple_brownian_fft.py
"""

from cddm import normalize, k_select, ccorr
from cddm.core import ccorr2, cmean

from cddm.core import normalize_ccorr

import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#setting this to 2 shows progress bar
conf.set_verbose(2)

np.random.seed(0)

t1 = np.load("simple_brownian_cddm_t1.npy")
t2 = np.load("simple_brownian_cddm_t2.npy")

v1 = np.load("simple_brownian_cddm_fft_0.npy")
v2 = np.load("simple_brownian_cddm_fft_1.npy")
v1 = v1/(v1[...,0,0][:,None,None])
v2 = v2/(v2[...,0,0][:,None,None])
#
#v1 = v1+ np.random.randn(64,33)
#v2 = v2+ np.random.randn(64,33)

v1 = v1/v1[...,0,0].mean()
v2 = v2/v2[...,0,0].mean()

#v1 = v1 - v1.mean(0)
#v2 = v2 - v2.mean(0)

m = (t1 == t2)
tmp = (v1[m].real)**2 + (v1[m].imag)**2 + (v2[m].real)**2 + (v2[m].imag)**2 

background = v1.mean(0), v2.mean(0)
bg1, bg2 = background

nframes = len(v1)

data1 = ccorr(v1,v2 , t1,t2, n = 2**10, norm = 0)
data2 = ccorr(v1,v2 , t1,t2, n = 2**10, norm = 1)

background = v1.mean(0), v2.mean(0)
#


data = normalize_ccorr(data1, background)



i,j = 4,8

plt.figure()

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)
x = np.arange(data.shape[-1])
plt.semilogx(x[1:], data[i,j][1:], "o")


np.save("simple_brownian_ccorr_linear.npy", data)

##now let us do some k-averaging
kdata = k_select(data, phi = 0, sector = 180, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    plt.semilogx(x[1:], c[1:], label = k)
plt.legend()

data = normalize_ccorr(data2, background)



i,j = 4,8

plt.figure()

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)
x = np.arange(data.shape[-1])
plt.semilogx(x[1:], data[i,j][1:], "o")


np.save("simple_brownian_ccorr_linear2.npy", data)

##now let us do some k-averaging
kdata = k_select(data, phi = 0, sector = 180, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    plt.semilogx(x[1:], c[1:], label = k)
plt.legend()




