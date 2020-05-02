"""
Demonstrates how to combine norm 2 and norm 3 data. This is how normalization
with norm = 6 or norm = 7 work. It takes norm 3 data, performs data denoising, clipping
and makes correlation data decreasing. Then it calculates weighting function
and perfomrs weighted average.

First, you must run:
    
$ cross_correlate_multi_live.py 
"""
from cddm.avg import weight_from_data, weighted_sum, decreasing, denoise
from cddm.multitau import log_merge, t_multilevel
from examples.conf import PERIOD, DATA_PATH
import os.path as p

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np

#: norm = 2 data normalized with scale = True
lin_2 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_2.npy"))
multi_2 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_2.npy"))

#: norm = 3 data normalized with scale = True
lin_3 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_3.npy"))
multi_3 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_3.npy"))

#: take (i,j) k value
(i,j) = (28,2)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

x,y = log_merge(lin_3, multi_3)

#: denoised data, used as a estimator for the weight
yd = denoise(decreasing(np.clip(denoise(y),0,1)))

ax2.semilogx(x,yd[i,j], label = "denoised")

x,y = log_merge(lin_2, multi_2)
ax2.semilogx(x,y[i,j], label = "norm 2")

x,y = log_merge(lin_3, multi_3)
ax2.semilogx(x,y[i,j], label = "norm 3")

#: now calculate weights and do weighted sum.
for norm in (6,7):
    #: x values for the interpolator
    x_lin = np.arange(lin_3.shape[-1])
    #: interpolated weight for linear part. We have already filtered the data, so pre_filter = False
    w_lin = weight_from_data(x_lin, x, yd, norm = norm, pre_filter = False)
    #: weighted correlation
    lin = weighted_sum(lin_2,lin_3,w_lin)
    
    #: obtain data points x-values for interpolator
    x_multi = t_multilevel(multi_3.shape, period = PERIOD)
    #: weight for multilevel part. We have already filtered the data, so pre_filter = False
    w_multi = weight_from_data(x_multi, x, yd, norm = norm, pre_filter = False)
    #: weighted correlation
    multi = weighted_sum(multi_2,multi_3,w_multi)
    #: plot weight
    xm,w = log_merge(w_lin, w_multi)
    ax1.semilogx(xm[1:],w[i,j,1:], label = "norm {}".format(norm))
    #: plot weighted sum data
    xm,wy = log_merge(lin, multi)
    ax2.semilogx(xm,wy[i,j], label = "norm {}".format(norm))


ax1.set_title("Weight (norm 2 data) @ k = ({},{})".format(i,j))
ax1.set_ylabel("")
ax1.legend()

ax2.set_title("Correlation data @ k = ({},{})".format(i,j))
ax2.set_xlabel("t")
ax2.set_ylabel("G / Var")
ax2.legend()
plt.show()


