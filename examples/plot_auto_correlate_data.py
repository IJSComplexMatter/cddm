"""
Auto-correlation plot example. You must first run

$ auto_correlate.py 
"""
import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np
x = np.load("auto_correlate_t.npy")
y = np.load("auto_correlate_data.npy")

for (i,j) in ((0,15),(-6,26), (6,26)):
    plt.semilogx(x,y[i,j], label =  "k = ({}, {})".format(i,j) )
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()
