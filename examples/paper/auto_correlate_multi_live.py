"""
For testing, inspects videos and performs live correlation calculation...
"""
from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop, show_video, play_threaded, asarrays, load, show_diff
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iacorr_multi, normalize_multi, log_merge, iccorr_multi
import cddm.conf

import numpy as np
from examples.paper.conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX, DATA_PATH, NFRAMES_STANDARD

#: see video_simulator for details, loads sample video
#import examples.paper.simple_video.dual_video as dual_video_simulator
import examples.paper.flow_video.fast_video as video_simulator
import importlib
importlib.reload(video_simulator) #recreates iterator

cddm.conf.set_showlib("pyqtgraph")


#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,),)*NFRAMES_STANDARD

video = video_simulator.video
#video = load(video,count = NFRAMES)

#video = show_diff(video, dt = (5,7,8,9,10), t1= t1, t2 = t2)
video = show_video(video)

#:perform the actual multiplication
video = multiply(video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax = KIMAX, kjmax = KJMAX)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

fft = play_threaded(fft)

if __name__ == "__main__":
    import os.path as p

    #we will show live calculation with the viewer
    viewer = MultitauViewer(scale = True)
    
    #initial mask parameters
    viewer.k = 15
    viewer.sector = 30
    
    fft = ((f,f) for (f,) in fft)
    
    
    #: now perform auto correlation calculation with default parameters and show live
    data, bg, var = iccorr_multi(fft, count = NFRAMES_STANDARD, viewer = viewer,complex = True)


    viewer.show()
