"""
Demonstrates how to compute cross- correlation function with the 
out-of-memory version of the multitau algorithm on masked data.
"""

from cddm.video import mask
from cddm.multitau import iccorr_multi
from cddm.viewer import MultitauViewer

from examples.mask_array import mask as m
import examples.cross_correlate_multi_live as cross_correlate_multi_live 
import importlib
importlib.reload(cross_correlate_multi_live) #recreates fft iterator

t1,t2 = cross_correlate_multi_live.t1, cross_correlate_multi_live.t2
fft = cross_correlate_multi_live.fft

fft_masked = mask(fft, mask = m)

data, bg, var = iccorr_multi(fft_masked, t1, t2, period = cross_correlate_multi_live.PERIOD)

#: or this
#data, bg, var = iccorr_multi(fft, t1, t2, period = cross_correlate_multi_live.PERIOD, 
#                             mask = m)

#: inspect the data
viewer = MultitauViewer(scale = True, mask = m)
viewer.set_data(data, bg, var)
viewer.set_mask(k = 25, angle = 0, sector = 180)
viewer.plot()
viewer.show()

