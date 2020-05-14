"""Standrad (constant FPS video)
"""
from cddm.sim import simple_brownian_video, seed, adc
from cddm.viewer import VideoViewer 
from cddm.video import load, crop, multiply
from examples.paper.conf import NFRAMES, SIMSHAPE, BACKGROUND, DELTA, INTENSITY, SIGMA, SHAPE,DUST1_PATH
import matplotlib.pyplot as plt

import numpy as np

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(range(NFRAMES), shape = SIMSHAPE,background = BACKGROUND,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY, dtype = "uint16")


#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

#: apply dust particles
dust = plt.imread(DUST1_PATH)[...,0] #float normalized to (0,1)
dust = ((dust,),)*NFRAMES

video = multiply(video, dust)

video = (tuple((adc(f, bit_depth = "12bit") for f in frames)) for frames in video)


if __name__ == "__main__":
    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below.
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: VideoViewer either expects a multi_frame iterator, or a numpy array
    viewer = VideoViewer(video, count = NFRAMES, vmin = 0, cmap = "gray", vmax = 4096)
    viewer.show()