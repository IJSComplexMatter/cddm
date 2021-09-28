"""

$ python flow_cross_correlate.py 

"""

import matplotlib.pyplot as plt
import numpy as np
from cddm.multitau import log_merge,  multilevel, merge_multilevel, log_average
import os.path as p

from examples.paper.conf import NFRAMES, DATA_PATH, SAVE_FIGS

#MARKERS = ["1", "2", "3",  "4", "+", "x"]
#TITLES = ["C-DDM ($q={}$)", "F-DDM ($q={}$)"]

#which K to plot
KI = 0

axs = plt.subplot(311), plt.subplot(312), plt.subplot(313)

NORM = (5,6,7)
NORM_LABEL = ("standard", "structured", "weighted")

NORM = (5,7)
NORM_LABEL = ("standard",  "weighted")


TMAX = 400
#NORM = (5,7)
#NORM_LABEL = ("standard", "weighted")

for j,norm in enumerate(NORM):
    fname = "flow_corr_dual_linear_norm{}.npy".format(norm)
    data = np.load(p.join(DATA_PATH, fname))[...,0:NFRAMES]
    KJ = 8
    y = data[KI,KJ]
    x = np.arange(len(y))
    

    ax = axs[0]
    
    
    ax.plot(x[:TMAX],y[:TMAX].real,"-", label = NORM_LABEL[j], fillstyle = "none")
    ax.set_ylabel("$\Re(C)$")
    ax = axs[1]
    
    
    ax.plot(x[:TMAX],y[:TMAX].imag,"-", label = NORM_LABEL[j], fillstyle = "none")
    ax.set_ylabel("$\Im(C)$")
    ax = axs[2]
    
    
    ax.plot(x[:TMAX],np.abs(y[:TMAX]),"-", label = NORM_LABEL[j], fillstyle = "none")
    ax.set_ylabel("$|C|$")
    

    
    #y_multi = multilevel(y,binning = True)
    #x_multi = multilevel(x,binning = True)
    
    #for i, (x,y) in enumerate(zip(x_multi, y_multi)):
    #    ax.semilogx(x[1:],y[1:], marker = MARKERS[i%6], linestyle = "-", label = "level {}".format(i))
    
    #x, y = merge_multilevel(y_multi)
    
    #ax.semilogx(x[1:],y[1:],"k", label = "log")
    #ax.set_title(TITLES[j].format(KI))
    ax.set_xlabel(r"$\tau$")
    #ax.set_ylabel(r"$g$")
    #ax.set_ylim(-0.2,1)

plt.legend()

plt.tight_layout()



if SAVE_FIGS:
    plt.savefig("plots/plot_flow.pdf")
    
plt.show()


