"""Plots cross-correlation and auto-correlation function
You must first create data by running 

$ cross_correlate.py 

and 

$ auto_correlate_fast.py
"""

import matplotlib.pyplot as plt
import numpy as np
from cddm.multitau import log_merge,  multilevel, merge_multilevel, log_average
import os.path as p

from examples.paper.conf import NFRAMES, DATA_PATH, SAVE_FIGS

MARKERS = ["1", "2", "3",  "4", "+", "x"]
TITLES = ["C-DDM ($q={}$)", "F-DDM ($q={}$)"]

#which K to plot
KI = 30

axs = plt.subplot(121), plt.subplot(122)


for j,fname in enumerate(("corr_dual_linear.npy","corr_fast_linear.npy")):
    
    data = np.load(p.join(DATA_PATH, fname))[...,0:NFRAMES]
    KJ = 0
    y = data[KI,KJ]
    x = np.arange(len(y))
    

    ax = axs[j]
    
    
    ax.semilogx(x[1:],y[1:],"-", label = "linear", fillstyle = "none")
    
    y_multi = multilevel(y,binning = True)
    x_multi = multilevel(x,binning = True)
    
    for i, (x,y) in enumerate(zip(x_multi, y_multi)):
        ax.semilogx(x[1:],y[1:], marker = MARKERS[i%6], linestyle = "-", label = "level {}".format(i))
    
    x, y = merge_multilevel(y_multi)
    
    ax.semilogx(x[1:],y[1:],"k", label = "log")
    ax.set_title(TITLES[j].format(KI))
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$g$")
    ax.set_ylim(-0.2,1)

plt.legend()

plt.tight_layout()



if SAVE_FIGS:
    plt.savefig("plots/plot_corr_example.pdf")
    
plt.show()


