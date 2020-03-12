"""
FFT tools. 

This module defines several functions for fft processing  of multi-frame data.
"""

import numpy as np
from cddm.video import VideoViewer, _FIGURES, play
from cddm.conf import CDDMConfig, MKL_FFT_INSTALLED, SCIPY_INSTALLED 

#from multiprocessing.pool import ThreadPool

if MKL_FFT_INSTALLED == True:
    import mkl_fft
    
if SCIPY_INSTALLED == True:
    import scipy.fftpack as spfft

class FFTViewer(VideoViewer):
    def __init__(self, title = "fft"):
        self.title = title
    
    def _prepare_image(self,im):
        im = np.fft.fftshift(np.log(1+np.abs(im)),0)
        return im/im.max()

def _window_title(name):
    i = len(_FIGURES)+1
    return "Fig.{}: {}".format(i, name)

def show_fft(video, id = 0, title = None):
    """Show """
    if title is None:
        title = _window_title("fft - camera {}".format(id))
    viewer = FFTViewer(title)
    
    for frames in video:
        _FIGURES[title] = (viewer, frames[id])
        yield frames
        
def show_alignment_and_focus(video, id = 0, title = None, clipfactor=0.1):
    '''To be changed'''
    
    if title is None:
        title = _window_title("fft - camera {}".format(id))
    title2="alignment"
    
    viewer1 = VideoViewer(title)
    viewer2 = VideoViewer(title2)
    
    
    for i,frames in enumerate(video):
        
        if i==0:
            f0=np.ones(np.shape(frames[id]))
            
        f=mkl_fft.fft2(frames[id])
        f=np.abs(f)
        f=np.abs((f-f0)/(256**2))
        f.clip(0,clipfactor)
        f=f/clipfactor
        f=np.fft.fftshift(f)
        
        im1=frames[0]
        im2=frames[1]
        diff=(im2/im2.mean())-(im1/im1.mean())+0.5
        
        _FIGURES[title] = (viewer1, f)
        _FIGURES[title2] = (viewer2, diff)
        
        f0=f.copy()
        
        yield frames
        
def _determine_cutoff_indices(shape, kisize = None, kjsize= None):
    if kisize is None:
        kisize = shape[0]
    else:    
        kisize = min(kisize, shape[0])
    if kjsize is None:
        kjsize = shape[1]
    else:
        kjsize = min(kjsize, shape[1])
    
    jstop = kjsize//2+1
    istop = kisize//2+1
    
    shape = kisize, jstop
    return shape, istop, jstop
    
def _rfft2(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    cutoff = a.shape[-1]//2 + 1
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "scipy":
        return spfft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "numpy":
        return np.fft.rfft2(a.real) #force real in case input is complex
    else:#default implementation is numpy fft2
        return np.fft.fft2(a)[...,0:cutoff]     

#def _rfft2_sequence(a, overwrite_x = False):
#    pool = ThreadPool(2)
#    workers = [pool.apply_async(_rfft2, args = (d, overwrite_x)) for d in a] 
#    results = [w.get() for w in workers]
#    pool.close()
#    return results


def rfft2(video, kisize = None, kjsize = None, overwrite_x = False):
    """A generator that performs rfft2 on a sequence of multi-frame data.
    """
    
    def f(frame, shape, istop, jstop):
        data = _rfft2(frame, overwrite_x = overwrite_x)
        vid = np.empty(shape,data.dtype)
        vid[:istop,:] = data[:istop,:jstop] 
        vid[-istop:,:] = data[-istop:,:jstop] 
        return vid
    
    if kisize is None and kjsize is None:
        #just do fft, no cropping
        for frames in video:
            #yield _rfft2_sequence(frames, overwrite_x = overwrite_x)
            yield tuple((_rfft2(frame, overwrite_x = overwrite_x) for frame in frames))
    else:
        #do fft with cropping
        out = None
        for frames in video:
            if out is None:
                shape = frames[0].shape
                shape, istop, jstop = _determine_cutoff_indices(shape, kisize, kjsize)
            out = tuple((f(frame,shape,istop,jstop) for frame in frames))
            yield out

        
if __name__ == "__main__":
    from cddm.video import random_dual_frame_video
    video = random_dual_frame_video()
    fft = rfft2(video)
    fft = show_fft(fft,0)

#    for frames in play(fft, fps = 20):
#        pass
