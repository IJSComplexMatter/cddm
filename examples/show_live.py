"""
Demonstrates how to perform live view of difference data, video, and fft of
dual-frame video.

Visualization takes place during the iteration over the iterator contructed
with the play function. Note that you can use the video for further processing
(correlation analysis) 

"""
from conf import SHAPE, NFRAMES, BACKGROUND

from cddm.video import  show_video, show_fft, play, show_diff, multiply
from cddm.sim import simple_brownian_video
from cddm.conf import set_showlib

import matplotlib.pyplot as plt

# test dual camera video (regular spaced)
video = simple_brownian_video(range(NFRAMES), range(NFRAMES), shape = SHAPE, background = BACKGROUND)

#: apply dust particles
dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
dust2 = plt.imread('dust2.png')[...,0]
dust = ((dust1,dust2),)* NFRAMES
video = multiply(video, dust)

video = show_video(video)
video = show_diff(video)
video = show_fft(video, mode ="real")

#: set fps to your required FPS. Video will be updated only if visualization
#: is fast enough not to interfere with the acquisition.
#: here video is again a valid video iterator, no visualization has yet took place
video = play(video, fps = 100)

#: you should use either cv2 or pyqtgraph, matplotlib is too slow
#set_showlib("cv2")
set_showlib("pyqtgraph")

# now go through frames and show videos
for frames in video:
    pass
