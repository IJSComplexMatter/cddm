"""
Builds sample dual-camera video of two-component system with two different
particles for two-exponent data fitting examples.
"""

from cddm.sim import simple_brownian_video, create_random_times1, adc
from cddm.viewer import VideoViewer 
from cddm.video import multiply, load, crop, add
import matplotlib.pyplot as plt
import numpy as np

# uppercase values
from examples.two_component.conf import NFRAMES, N_PARAMETER, SIMSHAPE, BACKGROUND, DELTA1,DELTA2, \
    INTENSITY1, INTENSITY2, SIGMA1,SIGMA2, SHAPE, DUST1_PATH, DUST2_PATH, SATURATION, ADC_BIT_DEPTH, NOISE_MODEL, READOUT_NOISE

#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(NFRAMES,n = N_PARAMETER)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video1 = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = BACKGROUND,num_particles = 10,
                              sigma = SIGMA1, delta = DELTA1, intensity = INTENSITY1, dtype = "uint16")

video2 = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = 0 ,num_particles = 100,
                              sigma = SIGMA2, delta = DELTA2, intensity = INTENSITY2, dtype = "uint16")

video = add(video1,video2)

#video = ((np.ones(SIMSHAPE,dtype = "uint16")*2**8,)*2 for i in range(NFRAMES))

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

#: apply dust particles
dust1 = plt.imread(DUST1_PATH)[...,0] #float normalized to (0,1)
dust2 = plt.imread(DUST2_PATH)[...,0]
dust = ((dust1,dust2),)*NFRAMES

video = multiply(video, dust)


video = (tuple((adc(f, noise_model = NOISE_MODEL, saturation = SATURATION, readout_noise = READOUT_NOISE, bit_depth = ADC_BIT_DEPTH) for f in frames)) for frames in video)


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