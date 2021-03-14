"""
Demonstrates the difference between various normalizations. 
You must first run

$ cross_correlate_multi_live.py 
"""
import matplotlib.pyplot as plt
import numpy as np

from examples.conf import DATA_PATH
import os.path as p

i,j = 4,15

#: plot the results
for norm in (1,2,3,5,6,7):
    x = np.load(p.join(DATA_PATH,"cross_correlate_multi_t.npy".format(norm)))
    y = np.load(p.join(DATA_PATH,"cross_correlate_multi_norm_{}_data.npy".format(norm)))
    plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )

plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()
