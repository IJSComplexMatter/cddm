#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""

from mask_array import mask
from cddm.multitau import iccorr_multi
from cddm.viewer import MultitauViewer
from cross_correlate_multi_live import fft, NFRAMES, t1,t2, PERIOD

import cddm.conf
cddm.conf.set_verbose(2)

data, bg, var = iccorr_multi(fft, t1, t2, period = PERIOD, level_size = 32, mask = mask)

#: inspect the data
viewer = MultitauViewer(scale = True, mask = mask)
viewer.set_data(data, bg, var)
viewer.set_mask(k = 25, angle = 0, sector = 180)
viewer.plot()
viewer.show()

