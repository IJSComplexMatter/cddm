#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:34:40 2020

@author: andrej
"""

from simple_brownian_video import get_dual_video,NFRAMES_DUAL
from cddm.viewer import VideoViewer 
from cddm.video import asarrays

video = get_dual_video()

#uncomment this if you want to load dual_video into two arrays for the first and the second sequence
#video,video2 = asarrays(video, count = NFRAMES_DUAL)

##VideoViewer either expects a multi_frame iterator, or a numpy array
viewer = VideoViewer(video, count = NFRAMES_DUAL, id = 0)
viewer.show()

