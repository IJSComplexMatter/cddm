"""
In this example we use a simulated single-camera video of o brownian motion
of spherical particles and perform multiple-tau auto-correlation analysis 
with different two different normalizations:
    1. background subtraction
    2. background subtraction + compensation

You must first create FFTs of the videos by calling

$ simple_brownian_video.py
$ simple_brownian_fft.py
"""

from cddm.core import subtract_background, stats, NORM_COMPENSATED, NORM_BASELINE
from cddm.multitau import log_merge, normalize_multi, acorr_multi
from cddm.viewer import MultitauViewer

#simulated diffusion constant 
from simple_brownian_video import D

import matplotlib.pyplot as plt
from cddm import conf
import numpy as np

#print mesages
conf.set_verbose(2)

#fourier transform of the video, axis = 0 (vid[0] is the first 2D frame)
vid = np.load("simple_brownian_ddm_fft.npy")

#remove background inplace. This improves the computed correlation function.
vid = subtract_background(vid, axis = 0, out = vid)
#compute cross-correlation function (max delay of 4096, now with norm == 1)
#we are using thread_divisor here
data = acorr_multi(vid, n = 16, axis = 0, norm = NORM_COMPENSATED, thread_divisor = 8)

#variance is needed for scaling. This function computes background and variance
bg, var = stats(vid)

# standard normalization norm = 0
data_lin0 = normalize_multi(data,background = bg, variance =var, scale = True, norm = NORM_BASELINE)

# compensated normalization (same as cdiff). Because we have computed  correlation
# with norm == 1 we can normalize 
data_lin1 = normalize_multi(data,background = bg, variance =var, scale = True, norm = NORM_COMPENSATED)

#log average, make data log-spaced
x_log, data_log0 = log_merge(*data_lin0)
x_log, data_log1 = log_merge(*data_lin1)

##dump  log_data to disk
np.save("simple_brownian_acorr_norm0_x.npy", x_log)
np.save("simple_brownian_acorr_norm0_data.npy", data_log0)
np.save("simple_brownian_acorr_norm1_x.npy", x_log)
np.save("simple_brownian_acorr_norm1_data.npy", data_log1)

#linear data x values
x_lin = np.arange(len(vid))

def exp_decay(x,i,j,D):
    return np.exp(-D*(i**2+j**2)*x)

plt.figure()

for k in ((14,0), (8,2)):
    #take i-th and j-th wavevector
    i,j = k
    #plot all but first element (x = 0 cannot be plotted in semilog)
    plt.semilogx(x_log[1:], data_log0[i,j,1:],"-", label = "norm = 0, k = ({},{})".format(i,j))
    plt.semilogx(x_log[1:], data_log1[i,j,1:],"-", label = "norm = 1, k = ({},{})".format(i,j))
    plt.semilogx(x_log[1:], exp_decay(x_log[1:],i,j,D), ":", label = "true, k = ({},{})".format(i,j))

plt.xlabel("dt")
plt.ylabel("g/var")
plt.title("auto-correlation with different normalizations")

plt.legend()


# this is a simple data viewer that you can use to inspect data
# here we perform basic normalization with scaling
viewer = MultitauViewer(scale = True)
viewer.set_data(data, bg, var)
viewer.plot()
viewer.show()