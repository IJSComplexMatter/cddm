"""
Demonstrates how to compute cross- correlation function with the 
out-of-memory version of the multitau algorithm with the mask parameter
"""

from mask_array import mask
from cddm.multitau import iccorr_multi
from cddm.viewer import MultitauViewer
import cross_correlate_multi_live
import importlib
importlib.reload(cross_correlate_multi_live) #recreates fft iterator

t1,t2 = cross_correlate_multi_live.t1, cross_correlate_multi_live.t2
fft = cross_correlate_multi_live.fft


import cddm.conf
cddm.conf.set_verbose(2)

data, bg, var = iccorr_multi(fft, t1, t2, period = cross_correlate_multi_live.PERIOD, 
                             level_size = 32, mask = mask)

#: inspect the data
viewer = MultitauViewer(scale = True, mask = mask)
viewer.set_data(data, bg, var)
viewer.set_mask(k = 25, angle = 0, sector = 180)
viewer.plot()
viewer.show()

