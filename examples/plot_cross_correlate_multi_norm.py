"""
Demonstrates the difference between different normalizations. 
You must first run

$ cross_correlate_multi_live.py 
"""
import matplotlib.pyplot as plt
import numpy as np

i,j = 4,15

#: plot the results
for norm in (0,1,2,3,6,7):
    x = np.load("cross_correlate_multi_norm_{}_t.npy".format(norm))
    y = np.load("cross_correlate_multi_norm_{}_data.npy".format(norm))
    plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )

plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()
