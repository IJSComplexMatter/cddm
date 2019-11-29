"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform cross-ddm analysis.

You must first create FFTs of the videos by calling simple_brownian_fft.py
"""

from cddm import normalize, k_select, cdiff

import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#setting this to 2 shows progress bar
conf.set_verbose(2)

SHAPE = (512, 512)

v1 = np.load("simple_brownian_cddm_fft_0.npy")
v2 = np.load("simple_brownian_cddm_fft_1.npy")

t1 = np.load("simple_brownian_cddm_t1.npy")
t2 = np.load("simple_brownian_cddm_t2.npy")

nframes = len(v1)

data,count = cdiff(v1,v2 , t1,t2, n = 2**10)

data = normalize((data,count))

i,j = 4,8

plt.figure()

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)
x = np.arange(data.shape[-1])
plt.semilogx(x[1:], data[i,j][1:], "o")


np.save("simple_brownian_cdiff_linear.npy", data)

##now let us do some k-averaging
kdata = k_select(data, phi = 0, sector = 180, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    plt.semilogx(x[1:], c[1:]/c[0], label = k)
plt.legend()
plt.show()



