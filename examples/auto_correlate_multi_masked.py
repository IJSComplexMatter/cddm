#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to compute fft of videos and the compute auto correlation
function with the out-of-memory version of the multitau algorithm.
"""

from mask_array import mask
from cddm.multitau import iacorr_multi
from cddm.viewer import MultitauViewer
from auto_correlate_multi import fft, NFRAMES

import cddm.conf
cddm.conf.set_verbose(2)

data, bg, var = iacorr_multi(fft,  count = NFRAMES, mask = mask)

#: inspect the data
viewer = MultitauViewer(scale = True, mask = mask)
viewer.set_data(data, bg, var)
viewer.set_mask(k = 25, angle = 0, sector = 30)
viewer.plot()
viewer.show()

