"""
Demonstrates how to use VideoViewer to inspect  dual camera video from frame 
iterator or list of data
"""

from cddm.sim import simple_brownian_video, create_random_times1
from cddm.viewer import VideoViewer 
from cddm.video import multiply
import matplotlib.pyplot as plt

nframes = 1024
#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(nframes,n = 16)

#: this creates a brownian motion frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame)
video = simple_brownian_video(t1,t2, shape = (512,512),background = 200)

#: apply dust particles
dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
dust2 = plt.imread('dust2.png')[...,0]
dust = ((dust1,dust2),)*nframes

video = multiply(video, dust)
#: no need to create list, but this way we load video into memory, and we can scroll 
#: back and forth with the viewer, uncomment the line below.
#video = list(video)

#: camera 1
viewer1 = VideoViewer(video, count = nframes, id = 0, vmin = 0, cmap = "gray")
viewer1.show()

#: camera 2
viewer2 = VideoViewer(video, count = nframes, id = 1, vmin = 0, cmap = "gray")
viewer2.show()