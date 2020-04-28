"""
Plots raw correlation data and demonstrates how to merge data. First you must
run

$ auto_correlate_multi.py 
"""
from cddm.multitau import log_merge
import matplotlib.pyplot as plt
import os.path as p
from examples.conf import DATA_PATH

#: load the normalized data to numpy files
import numpy as np
lin_data = np.load(p.join(DATA_PATH,"auto_correlate_multi_raw_fast.npy"))
multi_level = np.load(p.join(DATA_PATH,"auto_correlate_multi_raw_slow.npy"))

(i,j) = (4,12)

x,y = log_merge(lin_data, multi_level)    
plt.semilogx(x[1:],y[i,j,1:],"k-", label = "merged")

x = np.arange(lin_data.shape[-1])
plt.semilogx(x[1:], lin_data[i,j,1:], "o", fillstyle = "none", label = "linear (level 0)")

for n,data in enumerate(multi_level):
    x = x * 2
    plt.semilogx(x[1:], data[i,j,1:], "o", fillstyle = "none", label = "multi  (level {})".format(n+1))


plt.title("Multilevel data @ k = ({},{})".format(i,j))
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()


