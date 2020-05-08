"""Random triggering single camera video
"""
from cddm.sim import simple_brownian_video, seed
from cddm.viewer import VideoViewer 
from cddm.video import load, crop
from examples.paper.conf import NFRAMES, SIMSHAPE, BACKGROUND, DELTA, INTENSITY, SIGMA, SHAPE, PERIOD

import numpy as np

def create_random_times(isize = 1000, iperiod = PERIOD, nlow = 0):
    t0 = np.arange(isize)*iperiod
    r1 = np.random.randint(nlow, iperiod,  size = isize)
    return t0  +r1

t = create_random_times(NFRAMES)

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t, shape = SIMSHAPE,background = BACKGROUND,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY, dtype = "uint16")

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

video = (tuple((np.random.poisson(f) for f in frames)) for frames in video)

if __name__ == "__main__":
    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below.
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: VideoViewer either expects a multi_frame iterator, or a numpy array
    viewer = VideoViewer(video, count = NFRAMES,  cmap = "gray")
    viewer.show()