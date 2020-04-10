"""
Demonstrates how to use the computed correlation data. You must first run

$ auto_correlate_multi_live.py 
"""

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np
x = np.load("auto_correlate_multi_t.npy")
y = np.load("auto_correlate_multi_data.npy")

for (i,j) in ((4,12),(-6,16), (6,16)):
    plt.semilogx(x,y[i,j], label =  "k = ({}, {})".format(i,j) )
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()


