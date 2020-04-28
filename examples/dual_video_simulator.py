"""
Bulds sample dual-camera video and demonstrates how to use VideoViewer to inspect 
dual camera video from a frame iterator or list of data.
"""
from cddm.sim import simple_brownian_video, create_random_times1, seed
from cddm.viewer import VideoViewer 
from cddm.video import multiply, load, crop
import matplotlib.pyplot as plt

# uppercase values
from examples.conf import NFRAMES, N_PARAMETER, SIMSHAPE, BACKGROUND, DELTA, \
    INTENSITY, SIGMA, SHAPE,DUST1_PATH,DUST2_PATH

#: set seed for randum number generator, so that each run is the same
seed(0)

#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(NFRAMES,n = N_PARAMETER)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = BACKGROUND,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

#: apply dust particles
dust1 = plt.imread(DUST1_PATH)[...,0] #float normalized to (0,1)
dust2 = plt.imread(DUST2_PATH)[...,0]
dust = ((dust1,dust2),)*NFRAMES

video = multiply(video, dust)

if __name__ == "__main__":

    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: camera 1
    viewer1 = VideoViewer(video, count = NFRAMES, id = 0, vmin = 0, cmap = "gray")
    viewer1.show()
    
    #: camera 2
    viewer2 = VideoViewer(video, count = NFRAMES, id = 1, vmin = 0, cmap = "gray")
    viewer2.show()