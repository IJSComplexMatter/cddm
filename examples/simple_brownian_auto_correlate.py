"""
In this example we use a simulated single-camera video of a brownian motion
of spherical particles and perform auto-correlation analysis.

You must first create FFTs of the videos by calling

$ simple_brownian_video.py
$ simple_brownian_fft.py
"""

#tools needed
from cddm.core import normalize, acorr,subtract_background, stats
from cddm.viewer import DataViewer
from cddm.multitau import log_average 
from cddm import conf

#for plotting
import matplotlib.pyplot as plt
import numpy as np

#simulated diffusion constant 
from simple_brownian_video import D

#print mesages (level 2) set to 1 for less messages, or 0 for no messages
conf.set_verbose(2)

#fourier transform of the video, axis = 0 (vid[0] is the first 2D frame)
vid = np.load("simple_brownian_ddm_fft.npy")

#remove background inplace. This improves the computed correlation function.
vid = subtract_background(vid, axis = 0, out = vid)

#compute autocorrelation function (full size, using fft, standard normalization)
data = acorr(vid, axis = 0,  method = "fft", norm = 0)

#data is a length 5 tuple: thefirst element is the computed correlation, the second 
# is the number of counts, the third element is tau-sum of squares, fourth and 
#fifth are tau-sums and are defined if norm argument is greater than 0
#cor, count, _, _, _ = data

# another option is to compute directly (slow computation for full size analysis)
#data = acorr(vid, axis = 0, method = "corr", norm = 0)

#variance is needed for scaling. This function computes background and variance
bg, var = stats(vid, axis = 0)

#basic normalization with scaling. In this case we could have omitted background
# because it has already been subtracted from the video. Variance is needed for scaling.
data_lin = normalize(data, background = bg, variance = var, scale = True)

#log average, make data log-spaced with time doubling length of 16
x_log, data_log = log_average(data_lin, size = 16)

#dump  log_data to disk
np.save("simple_brownian_acorr_log_x.npy", x_log)
np.save("simple_brownian_acorr_log_data.npy", data_log)

#linear data x values
x_lin = np.arange(data_lin.shape[-1])

#theoretical correlation function
def exp_decay(x,i,j,D):
    return np.exp(-D*(i**2+j**2)*x)

for k in ((14,0), (24,2)):
    #take i-th and j-th wavevector
    i,j = k
    #plot all but first element (x = 0 cannot be plotted in semilog)
    plt.semilogx(x_lin[1:], data_lin[i,j,1:], "o",fillstyle = "none", label = "linear, k = ({},{})".format(i,j))
    plt.semilogx(x_log[1:], data_log[i,j,1:],"-", label = "log, k = ({},{})".format(i,j))
    plt.semilogx(x_log[1:], exp_decay(x_log[1:],i,j,D), ":", label = "true, k = ({},{})".format(i,j))

plt.xlabel("dt")
plt.ylabel("g/var")
plt.title("Normalized correlation")

plt.legend()
plt.show()

# this is a simple data viewer that you can use to inspect data
# here we perform basic normalization with scaling and log_average with size 10
viewer = DataViewer(scale = True, size = 10)
viewer.set_data(data, bg, var)
viewer.plot()
viewer.show()
