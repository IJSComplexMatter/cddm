"""
For testing, inspects videos and performs live correlation calculation...
"""
from cddm.viewer import MultitauViewer, CorrViewer, MultitauArrayViewer
from cddm.video import multiply, normalize_video, crop, show_video, play_threaded, asarrays, load, show_diff, show_data
from cddm.run import run
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge, iccorr_multi_iter
from cddm.core import iccorr
import cddm.conf

import numpy as np
from examples.paper.conf import NFRAMES, PERIOD, SHAPE, KIMAX, KJMAX, DATA_PATH

#: see video_simulator for details, loads sample video
#import examples.paper.simple_video.dual_video as dual_video_simulator
import examples.paper.flow_video.dual_video as dual_video_simulator
import importlib
importlib.reload(dual_video_simulator) #recreates iterator

cddm.conf.set_showlib("cv2")

t1, t2 = dual_video_simulator.t1, dual_video_simulator.t2 

#: create window for multiplication...
window = blackman(SHAPE)

#: we must create a video of windows for multiplication
window_video = ((window,window),)*NFRAMES

video = dual_video_simulator.video
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


#fft =  PlayingVideo(fft)
#fft =  play_threaded(fft)

#fft = threaded(fft)

if __name__ == "__main__":
    import os.path as p

    #we will show live calculation with the viewer
    viewer = MultitauArrayViewer(scale = True, semilogx = True, axes = (0,1))
    #viewer = CorrViewer(scale = True, semilogx = False)
    
    #initial mask parameters
    #viewer.k = 15
    #viewer.sector = 30
    
    
    #: now perform auto correlation calculation with default parameters and show live
    data_iter = iccorr_multi_iter(fft, t1, t2, period = PERIOD, complex = True, binning = 0)
    data_iter = show_data(data_iter, viewer)
    data, bg, var = run(data_iter, fps = 10, spawn = True)
    
    #data, bg, var = iccorr(fft, t1, t2,n=256, viewer = viewer,complex = True)

    viewer.show()
