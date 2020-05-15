"""Random triggering single camera video
"""
from cddm.core import auto_count
from cddm.sim import simple_brownian_video, create_random_times, adc
from cddm.viewer import VideoViewer 
from cddm.video import crop, multiply
from examples.paper.conf import NFRAMES, SIMSHAPE, BACKGROUND, DELTA, INTENSITY, \
    NPARTICLES, SIGMA, SHAPE, DUST1_PATH
import matplotlib.pyplot as plt
import numpy as np


while True:
    #make valid random time (all passible times present) 
    t = create_random_times(NFRAMES)
    count = auto_count(t,NFRAMES)
    if np.all(count):
        break

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t, shape = SIMSHAPE,background = BACKGROUND, particles = NPARTICLES,
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
    viewer = VideoViewer(video, count = NFRAMES,  cmap = "gray", vmin = 0, vmax = 4096)
    viewer.show()