"""
Demonstrates the difference between different normalizations. 
You must first run

$ cross_correlate_multi_live.py 
"""

import matplotlib.pyplot as plt
#: load the normalized data to numpy files
import numpy as np
x_0 = np.load("cross_correlate_multi_norm_0_t.npy")
y_0 = np.load("cross_correlate_multi_norm_0_data.npy")
x_1 = np.load("cross_correlate_multi_norm_1_t.npy")
y_1 = np.load("cross_correlate_multi_norm_1_data.npy")
x_2 = np.load("cross_correlate_multi_norm_2_t.npy")
y_2 = np.load("cross_correlate_multi_norm_2_data.npy")
x_3 = np.load("cross_correlate_multi_norm_3_t.npy")
y_3 = np.load("cross_correlate_multi_norm_3_data.npy")

i,j = 4,15
    
plt.semilogx(x_0,y_0[i,j], label =  "norm = 0" )
plt.semilogx(x_1,y_1[i,j], label =  "norm = 1" )
plt.semilogx(x_2,y_2[i,j], label =  "norm = 2" )
plt.semilogx(x_3,y_3[i,j], label =  "norm = 3" )
plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()


