"""
Builds sample dual-camera video of two-component system with two different
particles for two-exponent data fitting examples.
"""

from cddm.sim import simple_brownian_video, create_random_times1, adc
from cddm._sim_nb import adc_12bit
from cddm.viewer import VideoViewer 
from cddm.video import multiply, load, crop, add
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd

from examples.paper.two_component.conf import NFRAMES_DUAL, N_PARAMETER, SIMSHAPE, BACKGROUND, DELTA1, DELTA2, VMAX, DT_DUAL,\
    INTENSITY1,INTENSITY2, SIGMA1,SIGMA2, SHAPE, DUST1_PATH, DUST2_PATH, SATURATION, BIT_DEPTH, NOISE_MODEL, READOUT_NOISE,  NUM_PARTICLES1,NUM_PARTICLES2, APPLY_DUST


SAVEFIG = False

def move_pixels(frames, ni = 10):
    f1,f2 = frames
    return nd.zoom(f1,1.04)[10:512+10,10:512+10], f2

#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(NFRAMES_DUAL,n = N_PARAMETER)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video1 = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = BACKGROUND,num_particles = NUM_PARTICLES1,dt = DT_DUAL,
                              sigma = SIGMA1, delta = DELTA1, intensity = INTENSITY1, dtype = "uint16")

video2 = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = 0,num_particles = NUM_PARTICLES2,dt = DT_DUAL,
                              sigma = SIGMA2, delta = DELTA2, intensity = INTENSITY2, dtype = "uint16")


video = add(video1,video2)

#video = (move_pixels(frames) for frames in video)

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

# apply dust particles
if APPLY_DUST:
      dust1 = plt.imread(DUST1_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
      dust2 = plt.imread(DUST2_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)

      dust = ((dust1,dust2),)*NFRAMES_DUAL
      video = multiply(video, dust, dtype ="uint16")
      
noise_model = (NOISE_MODEL, NOISE_MODEL)      
      
video = (tuple((adc(f, noise_model = noise_model[i], saturation = SATURATION, 
                    readout_noise = READOUT_NOISE, bit_depth = BIT_DEPTH) for i,f in enumerate(frames))) for frames in video)

#video = load(video, NFRAMES_DUAL)

if __name__ == "__main__":

    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below
    if SAVEFIG:
        video = load(video, NFRAMES_DUAL) # loads and displays progress bar
        import matplotlib.pyplot as plt
        plt.figure(figsize = (8,3))
        plt.subplot(121)
        plt.imshow(video[0][0], cmap = "gray")
        plt.colorbar()
        plt.yticks([])
        plt.xticks([])
        plt.title("Camera 1")
        plt.subplot(122)
        plt.imshow(video[0][1], cmap = "gray")
        plt.colorbar()
        plt.yticks([])
        plt.xticks([])
        plt.title("Camera 2")
        plt.savefig("frames.pdf")
    
    

    #: camera 1
    viewer1 = VideoViewer(video, count = NFRAMES_DUAL, id = 0, vmin = 0, cmap = "gray", vmax = VMAX)
    viewer1.show()
    
    #: camera 2
    viewer2 = VideoViewer(video, count = NFRAMES_DUAL, id = 1, vmin = 0, cmap = "gray", vmax = VMAX)
    viewer2.show()