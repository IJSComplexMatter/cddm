#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrates how to use VideoViewer to inspect video from frame iterator or
list of data
"""

from cddm.sim import simple_brownian_video
from cddm.viewer import VideoViewer 
from cddm.video import asarrays

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(range(1024), shape = (512,512))

#: no need to create list, but this way we load video into memory, and we can scroll 
#: back and forth with the viewer. Uncomment the line below.
#video = list(video)

#: another option is to load into numpy array
#video, = asarrays(video)

#: VideoViewer either expects a multi_frame iterator, or a numpy array
viewer = VideoViewer(video, count =1024)
viewer.show()