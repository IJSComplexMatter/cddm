"""
Plots raw correlation data and demonstrates how to merge data.

$ cross_correlate_multi_live.py 
"""
from cddm.multitau import log_merge
from examples.conf import PERIOD, DATA_PATH
import os.path as p

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np

lin_data = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_fast_norm_2.npy"))
multi_level = np.load(p.join(DATA_PATH,"cross_correlate_multi_raw_slow_norm_2.npy"))

(i,j) = (4,2)

x = np.arange(lin_data.shape[-1])
plt.semilogx(x[1:], lin_data[i,j,1:], label = "linear (level 0)")

x = np.arange(multi_level.shape[-1]) * PERIOD

for n,data in enumerate(multi_level):
    x = x * 2
    plt.semilogx(x[1:], data[i,j,1:], "o", fillstyle = "none", label = "multi  (level {})".format(n+1))

x,y = log_merge(lin_data, multi_level)    
plt.semilogx(x[1:],y[i,j,1:],"k-", label = "merged")

plt.title("Multilevel data @ k = ({},{})".format(i,j))
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()


