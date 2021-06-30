"""
Dual-camera random triggered simple brownian motion video.

This script also creates sample frames from both cameras (Fig 1 in the paper).
"""

from cddm.sim import simple_brownian_video, create_random_times1, adc
from cddm.viewer import VideoViewer 
from cddm.video import multiply, load, crop
import matplotlib.pyplot as plt
import numpy as np

from examples.paper.simple_video.conf import SAVE_FIGS, NFRAMES_DUAL, N_PARAMETER, SIMSHAPE, BACKGROUND, DELTA, VMAX, DT_DUAL,\
    INTENSITY, SIGMA, SHAPE, DUST1_PATH, DUST2_PATH, SATURATION, BIT_DEPTH, NOISE_MODEL, READOUT_NOISE,  NUM_PARTICLES, APPLY_DUST

#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(NFRAMES_DUAL,n = N_PARAMETER)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t1,t2, shape = SIMSHAPE,background = BACKGROUND,num_particles = NUM_PARTICLES,dt = DT_DUAL,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY, dtype = "uint16")


#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

# apply dust particles
if APPLY_DUST:
      dust1 = plt.imread(DUST1_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
      dust2 = plt.imread(DUST2_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
      
      dust = ((dust1,dust2),)*(NFRAMES_DUAL) 
      video = multiply(video, dust, dtype ="uint16")
      
noise_model = (NOISE_MODEL, NOISE_MODEL)      
      
video = (tuple((adc(f, noise_model = noise_model[i], saturation = SATURATION, 
                    readout_noise = READOUT_NOISE, bit_depth = BIT_DEPTH) for i,f in enumerate(frames))) for frames in video)

#video = load(video, NFRAMES_DUAL)

if __name__ == "__main__":

    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below
    video = load(video, NFRAMES_DUAL) # loads and displays progress bar
    if SAVE_FIGS == True:
    
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