"""Standard (constant low FPS video)"""

from cddm.sim import simple_brownian_video, adc
from cddm.viewer import video_viewer 
from cddm.video import load, crop, multiply, save_video
from examples.paper.simple_video.conf import NFRAMES_STANDARD, SIMSHAPE, BACKGROUND, DELTA, DT_STANDARD, \
    INTENSITY, SIGMA, SHAPE,DUST1_PATH, BIT_DEPTH, VMAX, NOISE_MODEL, SATURATION, READOUT_NOISE, APPLY_DUST
import matplotlib.pyplot as plt

#: this cretaes a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(range(NFRAMES_STANDARD), shape = SIMSHAPE,background = BACKGROUND, dt = DT_STANDARD,
                              sigma = SIGMA, delta = DELTA, intensity = INTENSITY, dtype = "uint16")

#: crop video to selected region of interest 
video = crop(video, roi = ((0,SHAPE[0]), (0,SHAPE[1])))

#: apply dust particles
if APPLY_DUST:
    dust = plt.imread(DUST1_PATH)[0:SHAPE[0],0:SHAPE[1],0] #float normalized to (0,1)
    dust = ((dust,),)*NFRAMES_STANDARD
    video = multiply(video, dust)

video = (tuple((adc(f, noise_model = NOISE_MODEL, saturation = SATURATION, readout_noise = READOUT_NOISE, bit_depth = BIT_DEPTH) for f in frames)) for frames in video)

if __name__ == "__main__":
    #: no need to load video, but this way we load video into memory, and we 
    #: can scroll back and forth with the viewer. Uncomment the line below.
    save_video("brownian_motion.npy",video,NFRAMES_STANDARD)
    #video = load(video, NFRAMES) # loads and displays progress bar

    #: VideoViewer either expects a multi_frame iterator, or a numpy array
    #viewer = video_viewer(video, count = NFRAMES_STANDARD, imshow_options = {"vmin" : 0, "cmap" : "gray", "vmax" : VMAX})
    #viewer.show()