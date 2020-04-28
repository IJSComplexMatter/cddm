"""
Builds sample video and demonstrates how to use VideoViewer to inspect 
the video from a frame iterator or list of data.
"""
from cddm.sim import simple_brownian_video, seed
from cddm.viewer import VideoViewer 
from cddm.video import load, crop
from examples.conf import NFRAMES, SIMSHAPE, BACKGROUND, DELTA, INTENSITY, SIGMA, SHAPE

#: set seed for randum number generator, so that each run is the same
seed(0)

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(range(NFRAMES), shape = SIMSHAPE,background = BACKGROUND,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

if __name__ == "__main__":
    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below.
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: VideoViewer either expects a multi_frame iterator, or a numpy array
    viewer = VideoViewer(video, count = NFRAMES, vmin = 0, cmap = "gray")
    viewer.show()