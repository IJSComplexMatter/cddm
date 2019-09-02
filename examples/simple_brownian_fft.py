"""
This script opens videos DDM and c-DDM, peforms FFT, crops FFTs and dumps to disk

First you must generate videos with simple_brownian_video.py
"""

from cddm.video import show_video, play, apply_window, fromarrays, asarrays, asmemmaps
from cddm.window import blackman
from cddm.fft import rfft2, show_fft 
from cddm import normalize, k_select, acorr

import matplotlib.pyplot as plt

from cddm import conf
import numpy as np

#setting this to 2 shows progress bar
conf.set_verbose(2)

SHAPE = (512, 512)

vid = np.load("simple_brownian_ddm_video.npy")
nframes = len(vid)

#obtain frames iterator
video = fromarrays((vid,))
##apply blackman window 
window = blackman(SHAPE)
video = apply_window(video, (window,))
#perform rfft2 and crop data
video = rfft2(video, kisize = 64, kjsize = 64)
#load all frames into numpy array
#video, = asmemmaps("brownian_single_camera_fft", video, nframes)

#compute and create numpy array
video, = asarrays(video, nframes)
np.save("simple_brownian_ddm_fft.npy", video)

v1 = np.load("simple_brownian_cddm_video_0.npy")
v2 = np.load("simple_brownian_cddm_video_1.npy")
nframes = len(v1)

#obtain frames iterator
video = fromarrays((v1,v2))
##apply blackman window 
window = blackman(SHAPE)
video = apply_window(video, (window,window))
#perform rfft2 and crop data
video = rfft2(video, kisize = 64, kjsize = 64)
#load all frames into numpy array
#video, = asmemmaps("brownian_single_camera_fft", video, nframes)

#compute and create numpy array
asmemmaps("simple_brownian_cddm_fft", video, nframes)