"""
Compares results of linear analysis and multiple tau analysis. You must first run

$ auto_correlate.py
$ auto_correlate_multi.py 
"""
#change CWD to this file's path
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np

x_log = np.load("auto_correlate_t.npy")
y_log = np.load("auto_correlate_data.npy")

x_multi = np.load("auto_correlate_multi_t.npy")
y_multi = np.load("auto_correlate_multi_data.npy")

for (i,j) in ((4,12),(-6,16)):
    plt.semilogx(x_log,y_log[i,j], label =  "averaged k = ({}, {})".format(i,j) )
    plt.semilogx(x_multi,y_multi[i,j], label =  "multitau k = ({}, {})".format(i,j) )
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()


