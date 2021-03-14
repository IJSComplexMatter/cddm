"""
Demonstrates how to compute the cross-correlation function with the 
out-of-memory version of the multitau algorithm and do automatic backround 
subtraction.
"""
from cddm.multitau import iccorr_multi, normalize_multi, log_merge
import matplotlib.pyplot as plt

from conf import PERIOD

import cross_correlate_multi_live
import importlib
importlib.reload(cross_correlate_multi_live) #recreates fft iterator

t1,t2 = cross_correlate_multi_live.t1, cross_correlate_multi_live.t2
fft = cross_correlate_multi_live.fft

#: now perform auto correlation calculation with default parameters and show live
data, bg, var = iccorr_multi(fft, t1, t2, period = PERIOD, 
                             chunk_size = 128, auto_background = True)
i,j = 4,15

#: plot the results
for norm in (1,2,3,5,6,7):#,9,10,11,13,14,15):
    fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
    x,y = log_merge(fast, slow)
    plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )

plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()