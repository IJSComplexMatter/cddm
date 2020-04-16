"""
FFT tools. 

This module defines several functions for fft processing  of multi-frame data.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from cddm.conf import CDDMConfig, MKL_FFT_INSTALLED, SCIPY_INSTALLED, PYFFTW_INSTALLED,  detect_number_of_cores

#from multiprocessing.pool import ThreadPool

if MKL_FFT_INSTALLED == True:
    import mkl_fft
    
if SCIPY_INSTALLED == True:
    import scipy.fftpack as spfft
    
if PYFFTW_INSTALLED :
   import pyfftw.interfaces as fftw
   import pyfftw
   fftw.cache.enable()
   pyfftw.config.NUM_THREADS = detect_number_of_cores()



def _fft(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.fft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft(a, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return np.fft.fft(a) 
    elif libname == "pyfftw":
        return fftw.scipy_fftpack.fft(a, overwrite_x = overwrite_x)

def _ifft(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.ifft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.ifft(a, overwrite_x = overwrite_x)
    elif libname == "pyfftw":
        return fftw.scipy_fftpack.ifft(a, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return np.fft.ifft(a) 

#def _fft2(a, overwrite_x = False):
#    libname = CDDMConfig["fftlib"]
#    if libname == "mkl_fft":
#        return mkl_fft.fft2(a, overwrite_x = overwrite_x)
#    elif libname == "scipy":
#        return spfft.fft2(a, overwrite_x = overwrite_x)
#    elif libname == "numpy":
#        return np.fft.fft2(a) 
#    elif libname == "pyfftw":
#        return fftw.scipy_fftpack.fft2(a, overwrite_x = overwrite_x)
    
def _rfft2(a, overwrite_x = False):
    libname = CDDMConfig["rfft2lib"]
    cutoff = a.shape[-1]//2 + 1
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "scipy":
        return spfft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "numpy":
        return np.fft.rfft2(a.real) #force real in case input is complex
    elif libname == "pyfftw":
        return fftw.numpy_fft.rfft2(a.real) #force real in case input is complex 

        
def _determine_cutoff_indices(shape, kimax = None, kjmax= None):
    if kimax is None:
        kisize = shape[0]
        kimax = kisize//2 
    else:    
        kisize = kimax*2+1
        if kisize > (shape[0]//2)*2+1:
            raise ValueError("kimax too large for a given frame")
    if kjmax is None:
        kjmax = shape[1]//2 
    else:
        if kjmax > shape[1]//2:
            raise ValueError("kjmax too large for a given frame")
    
    istop = kimax+1
    jstop = kjmax+1
    
    shape = kisize, jstop
    return shape, istop, jstop

def rfft2(video, kimax = None, kjmax = None, overwrite_x = False):
    """A generator that performs rfft2 on a sequence of multi-frame data.
    
    Shape of the output depends on kimax and kjmax. It is (2*kimax+1, kjmax +1), 
    or same as the result of rfft2 if kimax and kjmax are not defined.
    
    Parameters
    ----------
    video : iterable
        An iterable of multi-frame data
    kimax : float, optional
        Max value of the wavenumber in vertical axis (i)
    kjmax : float, optional
        Max value of the wavenumber in horizontal axis (j)
    overwrite_x : bool, optional
        If input type is complex and fft library used is not numpy, fft can 
        be performed inplace to speed up computation.
        
    Returns
    -------
    video : iterator
        An iterator over FFT of the video.
    """
    
    def f(frame, shape, istop, jstop):
        data = _rfft2(frame, overwrite_x = overwrite_x)
        vid = np.empty(shape,data.dtype)
        vid[:istop,:] = data[:istop,:jstop] 
        vid[-istop+1:,:] = data[-istop+1:,:jstop] 
        return vid
    
    if kimax is None and kjmax is None:
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
                shape, istop, jstop = _determine_cutoff_indices(shape, kimax, kjmax)
            out = tuple((f(frame,shape,istop,jstop) for frame in frames))
            yield out

def normalize_fft(video, inplace = False, dtype = None):
    """Normalizes each frame in fft video to the mean value (intensity) of
    the [0,0] component of fft.
    
    Parameters
    ----------
    video : iterable
        Input multi-frame iterable object. Each element of the iterable is a tuple
        of ndarrays (frames)
    inplace : bool, optional
        Whether tranformation is performed inplace or not. 
    dtype : numpy dtype
        If specifed, determines output dtype. Only valid if inplace == False.
                
    Returns
    -------
    video : iterator
        A multi-frame iterator
    """
    for frames in video:
        if inplace == True:
            yield tuple((np.divide(frame, frame[0,0], frame) for frame in frames))
        else:
            yield tuple((np.asarray(frame / frame[0,0], dtype = dtype) for frame in frames))   
        
#if __name__ == "__main__":
#    import cddm.conf
#
#    from cddm.video import random_video, show_diff, show_video, show_fft, play
#    video = random_video(dual = True, shape = (512,256))
#    #video = show_video(video,0)
#    #video = show_diff(video)
#    video = show_fft(video,0)
#    #video = rfft2(video,63,63)
#    
#    
#    for frames in play(video, fps = 20):
#        pass
