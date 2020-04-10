"""
Demonstrates how to compute fft of videos and the compute cross-correlation
function with the out-of-memory version of the multitau algorithm and do
aoutomatic backround subtraction.
"""

from cddm.viewer import MultitauViewer
from cddm.video import multiply, normalize_video, crop
from cddm.window import blackman
from cddm.fft import rfft2, normalize_fft
from cddm.multitau import iccorr_multi, normalize_multi, log_merge
from cddm.sim import simple_brownian_video, create_random_times1
import matplotlib.pyplot as plt

nframes = 1024
n = 16
#random time according to Eq.7 from the SoftMatter paper
t1, t2 = create_random_times1(nframes,n = n)

#: this creates a brownian motion multi-frame iterator. 
#: each element of the iterator is a tuple holding a single numpy array (frame,)
video = simple_brownian_video(t1,t2, shape = (512+32,512+32))

#: crop video to selected region of interest 
video = crop(video, roi = ((0,512), slice(0,512)))

#: apply dust particles
dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
dust2 = plt.imread('dust2.png')[...,0]
dust = ((dust1,dust2),)*nframes
video = multiply(video, dust)

#: create window for multiplication...
window = blackman((512,512))

#: we must create a video of windows for multiplication
window_video = ((window,window),)*nframes

#:perform the actual multiplication
video = multiply(video, window_video)

#: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
#video = normalize_video(video)

#: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
fft = rfft2(video, kimax =37, kjmax = 37)

#: you can also normalize each frame with respect to the [0,0] component of the fft
#: this it therefore equivalent to  normalize_video
#fft = normalize_fft(fft)

#: now perform auto correlation calculation with default parameters and show live
data, bg, var = iccorr_multi(fft, t1, t2, period = 2*n, 
                             chunk_size = 128, auto_background = True)

i,j = 4,15

#: plot the results
for norm in (0,1,2,3):
    fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
    x,y = log_merge(fast, slow)
    plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )

plt.xlabel("t")
plt.ylabel("G / Var")
plt.legend()
plt.show()



