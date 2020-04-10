"""
This script opens videos DDM and c-DDM, peforms FFT, crops FFTs and dumps to disk

First you must generate videos with simple_brownian_video.py
"""

from cddm.video import  multiply, fromarrays, asarrays, asmemmaps
from cddm.window import blackman
from cddm.fft import rfft2


from cddm import conf
import numpy as np


def fft_video(vid, window = True):
    shape = vid[0].shape[1:]
    #obtain frames iterator
    video = fromarrays(vid)
    ##apply blackman window 
    if window == True:
        window = blackman(shape)
        video = multiply(video, ((window,)*len(vid),)*len(vid[0]))
    #perform rfft2 and crop data. it is best to use odd fo kjmax... so that
    # outputsize is even (32) in this case... and a power of two.
    video = rfft2(video, kimax = 31, kjmax = 31)
    return video

if __name__ == "__main__":  
    #setting this to 2 shows progress bar
    conf.set_verbose(2)
    
    vid = np.load("simple_brownian_ddm_video.npy")
    nframes = len(vid)
    #computes fft and returns fft iterator of single-frame data (viewed as multi-frame with a single element)
    fft = fft_video((vid,))
    #load all frames into numpy array
    fft, = asarrays(fft, nframes)
    np.save("simple_brownian_ddm_fft.npy", fft)
    
    v1 = np.load("simple_brownian_cddm_video_0.npy")
    v2 = np.load("simple_brownian_cddm_video_1.npy")
    nframes = len(v1)
    
    fft = fft_video((v1,v2))
    #compute and create numpy array
    asmemmaps("simple_brownian_cddm_fft", fft, nframes)