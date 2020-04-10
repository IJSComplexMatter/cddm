"""
FFT tools. 

This module defines several functions for fft processing  of multi-frame data.
"""

import numpy as np
from cddm.video import ImageShow, _FIGURES, play, figure_title
from cddm.conf import CDDMConfig, MKL_FFT_INSTALLED, SCIPY_INSTALLED
from queue import Queue

#from multiprocessing.pool import ThreadPool

if MKL_FFT_INSTALLED == True:
    import mkl_fft
    
if SCIPY_INSTALLED == True:
    import scipy.fftpack as spfft

def show_fft(video, id = 0, clip = None, title = None):
    """Show fft
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    id : int
        Frame index
    clip : float, optional
        Clipping value. If not given, it is determined automatically.
    title : str, optional
        Unique title of the video. You can use :func:`.video.figure_title`
        to create a unique name.
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    
    """
    if title is None:
        title = figure_title("fft - camera {}".format(id))
    viewer = ImageShow(title)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    for frames in video:
        if queue.empty():
            im = _clip_fft(frames[id],clip)
            queue.put(im, block = False)
        yield frames
        
def _clip_fft(im, clip = None):
    im = np.abs(im)
    if clip is None:
        im[0,0] = 0
        clip = im.max()
        
    im = im/clip
    im = im.clip(0,1)    
    return np.fft.fftshift(im,0)

def show_fftdiff(video, clip = None, title = None):
    """Show fft difference video
    
    Parameters
    ----------
    video : iterator
        A multi-frame iterator
    clip : float, optional
        Clipping value. If not given, it is determined automatically.
    title : str, optional
        Unique title of the video. You can use :func:`figure_title``
        a to produce unique name.
    
    Returns
    -------
    video : iterator
        A multi-frame iterator
    
    """
    if title is None:
        title = figure_title("FFT diff")
    viewer = ImageShow(title)
    queue = Queue(1)
    _FIGURES[title] = (viewer, queue)
    
    for frames in video:
        if queue.empty():
            im = frames[1]-frames[0]
            im = _clip_fft(im,clip)
            queue.put(im, block = False)
        yield frames
        

def _fft(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.fft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft(a, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return np.fft.fft(a) 
    else:#default implementation is numpy fft
        return np.fft.fft(a)

def _ifft(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.ifft(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.ifft(a, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return np.fft.ifft(a) 
    else:#default implementation is numpy ifft
        return np.fft.ifft(a)

def _fft2(a, overwrite_x = False):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, overwrite_x = overwrite_x)
    elif libname == "scipy":
        return spfft.fft2(a, overwrite_x = overwrite_x)
    elif libname == "numpy":
        return np.fft.fft2(a) 
    else:#default implementation is numpy fft2
        return np.fft.fft2(a)

    
def _rfft2(a, overwrite_x = False):
    libname = CDDMConfig["rfft2lib"]
    cutoff = a.shape[-1]//2 + 1
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "scipy":
        return spfft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "numpy":
        return np.fft.rfft2(a.real) #force real in case input is complex
    else:#default implementation is numpy fft2
        return np.fft.fft2(a)[...,0:cutoff]     

        
def _determine_cutoff_indices(shape, kimax = None, kjmax= None):
    if kimax is None:
        kisize = shape[0]
    else:    
        kisize = kimax*2+1
        if kisize >= shape[0]:
            raise ValueError("kimax too large for a given frame")
    if kjmax is None:
        kjsize = shape[1]
    else:
        kjsize = kjmax*2+1
        if kjsize > shape[1]:
            raise ValueError("kjmax too large for a given frame")
    
    jstop = kjsize//2+1
    istop = kisize//2+1
    
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
        vid[-istop:,:] = data[-istop:,:jstop] 
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


        
if __name__ == "__main__":
    import cddm.conf
    cddm.conf.set_fftlib("numpy")
    from cddm.video import random_video, show_diff, show_video
    video = random_video(dual = True)
    video = show_video(video,0)
    video = show_diff(video)
    video = rfft2(video,63,63)
    video = show_fft(video,0)
    
    for frames in play(video, fps = 20):
        pass
