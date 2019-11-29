"""
In this example we use a simulated single-camera video of o brownian motion
of spherical particles and perform ddm analysis.

You must first create FFTs of the videos by calling simple_brownian_fft.py
"""

from cddm import normalize, k_select, adiff

import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#setting this to 2 shows progress bar
conf.set_verbose(2)

SHAPE = (512, 512)

v = np.load("simple_brownian_ddm_fft.npy")


nframes = len(v)

data,count = adiff(v,n = 2**10)

data = normalize((data,count))

i,j = 4,8

plt.figure()

#plot fast data  at k =(i,j) and for x > 0 (all but first data point)
x = np.arange(data.shape[-1])
plt.semilogx(x[1:], data[i,j][1:], "o")


np.save("simple_brownian_adiff_linear.npy", data)

##now let us do some k-averaging
kdata = k_select(data, phi = 0, sector = 180, kstep = 1)

plt.figure()
#
for k, c in kdata: 
    plt.semilogx(x[1:], c[1:]/c[-1], label = k)
plt.legend()
plt.show()



