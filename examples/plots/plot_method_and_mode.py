"""
Demonstrates the use and equivalence of method and mode options
"""
from examples.auto_correlate import fft_array
from cddm.core import acorr, normalize, stats

import matplotlib.pyplot as plt

bg, var = stats(fft_array)

for method in ("corr","diff"):
    if method == "corr":
        data = acorr(fft_array, method = "fft") #fft,so that it is faster
    else:
        data = acorr(fft_array, method = "diff", n = 256)
    for mode in ("diff", "corr"):
        data_lin = normalize(data, bg, var, mode = mode, scale = True)
        plt.semilogx(data_lin[4,12], label = "mode = {}; method = {}".format(mode, method))

plt.legend()
plt.show()
    
