"""
Auto-correlation plot example. You must first run

$ auto_correlate.py 
"""
import matplotlib.pyplot as plt
from examples.conf import DATA_PATH
import os.path as p
#: load the normalized data to numpy files
import numpy as np
x = np.load(p.join(DATA_PATH,"auto_correlate_t.npy"))
y = np.load(p.join(DATA_PATH,"auto_correlate_data.npy"))

for (i,j) in ((0,15),(-6,26), (6,26)):
    plt.semilogx(x,y[i,j], label =  "k = ({}, {})".format(i,j) )
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()
