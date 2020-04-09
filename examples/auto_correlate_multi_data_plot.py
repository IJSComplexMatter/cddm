"""
Demonstrates how to use the computed correlation data. You must first run

$ auto_correlate_multi_live.py 
"""

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np
x = np.load("auto_correlat_multi_live_t.npy")
y = np.load("auto_correlat_multi_live_data.npy")

#plot correlation for k = (4,12)
plt.semilogx(x,y[4,12])
plt.xlabel("t")
plt.ylabel("G / Var")
plt.show()


