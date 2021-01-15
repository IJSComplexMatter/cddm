"""Random triggering single camera video
"""
from cddm.core import auto_count
from cddm.sim import simple_brownian_video, create_random_time, adc
from cddm.viewer import VideoViewer 
from cddm.video import crop, multiply
from examples.paper.one_component.conf import NFRAMES_RANDOM, SIMSHAPE, BACKGROUND, DELTA, INTENSITY, PERIOD_RANDOM,\
    NUM_PARTICLES, SIGMA, SHAPE, DUST1_PATH, BIT_DEPTH, VMAX, SATURATION, READOUT_NOISE, NOISE_MODEL, APPLY_DUST, DT_RANDOM, DT_STANDARD
import matplotlib.pyplot as plt
import numpy as np


while True:
    #make valid random time (all passible times present) 
    t = create_random_time(NFRAMES_RANDOM,n=PERIOD_RANDOM)
    count = auto_count(t,NFRAMES_RANDOM)
    
    #make sure all times are present, if so break, else try again
    if np.all(count):
        break

#t = create_random_time(NFRAMES_RANDOM, n=PERIOD, dt_min = PERIOD//2)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t, shape = SIMSHAPE,background = BACKGROUND, num_particles = NUM_PARTICLES, dt = DT_RANDOM,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY, dtype = "uint16")

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

#: apply dust particles
if APPLY_DUST:
    dust_frame = plt.imread(DUST1_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
    dust = ((dust_frame,),)*NFRAMES_RANDOM
    video = multiply(video, dust)

video = (tuple((adc(f, noise_model = NOISE_MODEL, saturation = SATURATION, readout_noise = READOUT_NOISE, bit_depth = BIT_DEPTH) for f in frames)) for frames in video)


if __name__ == "__main__":
    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below.
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: VideoViewer either expects a multi_frame iterator, or a numpy array
    viewer = VideoViewer(video, count = NFRAMES_RANDOM,  cmap = "gray", vmin = 0, vmax = VMAX)
    viewer.show()