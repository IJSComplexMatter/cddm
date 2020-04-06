"""
In this example we use a simulated dual-camera video of o brownian motion
of spherical particles and perform multiple-tau cross-correlation analysis 
with different two different normalizations:
    1. background subtraction
    2. background subtraction + compensation

You must first create FFTs of the videos by calling

$ simple_brownian_video.py
$ simple_brownian_fft.py
"""

from cddm.core import subtract_background, stats, NORM_COMPENSATED, NORM_BASELINE
from cddm.multitau import log_merge, normalize_multi, ccorr_multi
from cddm.video import fromarrays
from cddm.viewer import MultitauViewer

#simulated diffusion constant 
from simple_brownian_video import D, PERIOD

import matplotlib.pyplot as plt
from cddm import conf
import numpy as np

#print mesages
conf.set_verbose(2)

#fourier transform of the video, axis = 0 (vid[0] is the first 2D frame)
vid1 = np.load("simple_brownian_cddm_fft_0.npy")
vid2 = np.load("simple_brownian_cddm_fft_1.npy")

t1 = np.load("simple_brownian_cddm_t1.npy")
t2 = np.load("simple_brownian_cddm_t2.npy")

#remove background inplace. This improves the computed correlation function.
vid1 = subtract_background(vid1, axis = 0, out = vid1)
vid2 = subtract_background(vid2, axis = 0, out = vid2)

#compute cross-correlation function (max delay of 4096, now with norm == 1)
#we are using thread_divisor here
data = ccorr_multi(vid1,vid2,t1,t2, period = PERIOD, n = 16, axis = 0, norm = NORM_COMPENSATED, thread_divisor = 8)
v = fromarrays((vid1,vid2))

#variance is needed for scaling. This function computes background and variance
bg, var = stats(vid1,vid2)

# standard normalization norm = 0
data_lin0 = normalize_multi(data,background = bg, variance =var, scale = True, norm = NORM_BASELINE)

# compensated normalization (same as cdiff). Because we have computed  correlation
# with norm == 1 we can normalize 
data_lin1 = normalize_multi(data,background = bg, variance =var, scale = True, norm = NORM_COMPENSATED)

#log average, make data log-spaced
x_log, data_log0 = log_merge(*data_lin0)
x_log, data_log1 = log_merge(*data_lin1)

##dump  log_data to disk
np.save("simple_brownian_ccorr_norm0_x.npy", x_log)
np.save("simple_brownian_ccorr_norm0_data.npy", data_log0)
np.save("simple_brownian_ccorr_norm1_x.npy", x_log)
np.save("simple_brownian_ccorr_norm1_data.npy", data_log1)

#linear data x values
x_lin = np.arange(len(vid1))

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
plt.title("cross-correlation with different normalizations")

plt.legend()


# this is a simple data viewer that you can use to inspect data
# here we perform basic normalization with scaling
viewer = MultitauViewer(scale = True)
viewer.set_data(data, bg, var)
viewer.plot()
viewer.show()