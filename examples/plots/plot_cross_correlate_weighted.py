"""
Demonstrates how to combine norm 5 and norm 6 data. This is how normalization
with norm = 3 or norm = 7 work. It takes norm 6 data, performs data denoising, clipping
and makes correlation data decreasing. Then it calculates weighting function
and perfomrs weighted average.

First, you must run:
    
$ cross_correlate_multi_live.py 
"""
from cddm.avg import decreasing, denoise, log_interpolate
from cddm.norm import weight_from_data, weighted_sum
from cddm.multitau import log_merge, t_multilevel
from examples.conf import PERIOD, DATA_PATH
import os.path as p

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np

#: norm = 2 data normalized with scale = True
lin_5 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_5.npy"))
multi_5 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_5.npy"))

#: norm = 3 data normalized with scale = True
lin_6 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_6.npy"))
multi_6 = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_6.npy"))

#: take (i,j) k value
(i,j) = (12,2)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

x,y = log_merge(lin_6, multi_6)

#: denoised data, used as a estimator for the weight
yd = denoise(decreasing(np.clip(denoise(y),0,1)))

x,y = log_merge(lin_5, multi_5)
ax2.semilogx(x,y[i,j], label = "norm 5")

x,y = log_merge(lin_6, multi_6)
ax2.semilogx(x,y[i,j], label = "norm 6")

ax1.semilogx(x,yd[i,j], label = "g1")

#: now calculate weights and do weighted sum.

#: x values for the interpolator
x_lin = np.arange(lin_6.shape[-1])
#: iweight for linear part. We have already filtered the data, so pre_filter = False
w_lin = weight_from_data(yd, pre_filter = False)
#: interpolate data points to x_lin values using log-interpolator
w_lin= log_interpolate(x_lin, x, w_lin) 
#: weighted correlation
lin = weighted_sum(lin_6,lin_5,w_lin)

#: obtain data points x-values for the interpolator
x_multi = t_multilevel(multi_6.shape, period = PERIOD)
#: weight for multilevel part. We have already filtered the data, so pre_filter = False
w_multi = weight_from_data(yd,pre_filter = False)
#: interpolate data points to x_multi values using log-interpolator
w_multi = log_interpolate(x_multi, x, w_multi) 
#: weighted correlation
multi = weighted_sum(multi_6,multi_5,w_multi)
#: plot weight
xm,w = log_merge(w_lin, w_multi)
ax1.semilogx(xm[1:],w[i,j,1:], label = "w")
ax1.semilogx(xm[1:],1-w[i,j,1:], label = "1-w")
#: plot weighted sum data
xm,wy = log_merge(lin, multi)
ax2.semilogx(xm,wy[i,j], label = "weighted")


ax1.set_title("Weight @ k = ({},{})".format(i,j))
ax1.set_ylabel("")
ax1.legend()

ax2.set_title("Correlation data @ k = ({},{})".format(i,j))
ax2.set_xlabel("t")
ax2.set_ylabel("G / Var")
ax2.legend()
plt.show()


