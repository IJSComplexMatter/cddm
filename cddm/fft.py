"""
FFT tools. 

This module defines several functions for fft processing  of multi-frame data.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from cddm.conf import CDDMConfig, MKL_FFT_INSTALLED, SCIPY_INSTALLED, PYFFTW_INSTALLED,  detect_number_of_cores, CDTYPE
from cddm.decorators import deprecated

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

def _fft2(a, overwrite_x = False, extra = {}):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.fft2(a, overwrite_x = overwrite_x, **extra)
    elif libname == "scipy":
        return spfft.fft2(a, overwrite_x = overwrite_x, **extra)
    elif libname == "numpy":
        return np.fft.fft2(a, **extra) 
    elif libname == "pyfftw":
        return fftw.scipy_fftpack.fft2(a, overwrite_x = overwrite_x, **extra)

def _ifft2(a, overwrite_x = False, extra = {}):
    libname = CDDMConfig["fftlib"]
    if libname == "mkl_fft":
        return mkl_fft.ifft2(a, overwrite_x = overwrite_x, **extra)
    elif libname == "scipy":
        return spfft.ifft2(a, overwrite_x = overwrite_x,**extra)
    elif libname == "numpy":
        return np.fft.ifft2(a, **extra) 
    elif libname == "pyfftw":
        return fftw.scipy_fftpack.ifft2(a, overwrite_x = overwrite_x, **extra)
    
def _rfft2(a, overwrite_x = False, extra = {}):
    libname = CDDMConfig["rfft2lib"]
    cutoff = a.shape[-1]//2 + 1
    if libname == "mkl_fft":
        out = mkl_fft.rfftn_numpy(a.real,axes =(-2, -1), **extra)
        #return mkl_fft.fft2(a, overwrite_x = overwrite_x)[...,0:cutoff]
    elif libname == "scipy":
        out = spfft.fft2(a, overwrite_x = overwrite_x, **extra)[...,0:cutoff]
    elif libname == "numpy":
        out = np.fft.rfft2(a.real, **extra) #force real in case input is complex
    elif libname == "pyfftw":
        out = fftw.numpy_fft.rfft2(a.real, **extra) #force real in case input is complex 
    
    # depending on how the libraries are compiled, the output may not be of same dtype as requested
    # float32 may be converted to complex128... so we make sure it is of specified type.
    return np.asarray(out, CDTYPE)

def _determine_kimax(kisize, kimax):
    if kimax is None:
        kimax = kisize//2 
        return kimax, kisize
    else:    
        n = kimax*2+1
        if n > (kisize//2)*2+1:
            return None, None
        return kimax, kimax*2+1

def _determine_kjmax(kjsize, kjmax):
    if kjmax is None:
        kjmax = kjsize-1 
        return kjmax, kjsize
    else:
        if kjmax > kjsize-1:
            return None, None
        return kjmax, kjmax + 1

def _determine_cutoff_indices(shape, kimax = None, kjmax= None, mode = "rfft2"):
    kimax, kisize = _determine_kimax(shape[0],kimax)
    if mode == "rfft2":
        kjmax, kjsize = _determine_kjmax(shape[1],kjmax)
    elif mode == "fft2":
        kjmax, kjsize = _determine_kimax(shape[1],kjmax)
    else:
        raise ValueError("Invalid mode.")
    
    if kimax is None:
        raise ValueError("kimax too large for a given frame")
    if kjmax is None:
        raise ValueError("kjmax too large for a given frame")
    
    istop = kimax+1
    jstop = kjmax+1
    
    shape = kisize, kjsize
    return shape, istop, jstop

def rfft2_crop(x, kimax = None, kjmax = None):
    """Crops rfft2 data.
    
    Parameters
    ----------
    x : ndarray
        FFT2 data (as returned by np.rfft2 for instance). FFT2 must be over the
        last two axes.
    kimax : int, optional
        Max k value over the first (-2) axis of the FFT.
    kjmax : int, optional
        Max k value over the second (-1) axis of the FFT.   
        
    Returns
    -------
    out : ndarray
        Cropped fft array.
    """
    if kimax is None and kjmax is None:
        return x
    else:
        x = np.asarray(x)
        shape = x.shape[-2:]
        shape, istop, jstop = _determine_cutoff_indices(shape, kimax, kjmax, mode = "rfft2")
        
        out = np.empty(shape = x.shape[:-2] + shape, dtype = x.dtype)
        out[...,:istop,:] = x[...,:istop,:jstop] 
        out[...,-istop+1:,:] = x[...,-istop+1:,:jstop] 
        return out
    
def fft2_crop(x, kimax = None, kjmax = None):
    """Crops fft2 data.
    
    Parameters
    ----------
    x : ndarray
        FFT2 data (as returned by np.fft2 for instance). FFT2 must be over the
        last two axes.
    kimax : int, optional
        Max k value over the first (-2) axis of the FFT.
    kjmax : int, optional
        Max k value over the second (-1) axis of the FFT.   
        
    Returns
    -------
    out : ndarray
        Cropped fft array.
    """
    if kimax is None and kjmax is None:
        return x
    else:
        x = np.asarray(x)
        shape = x.shape[-2:]
        shape, istop, jstop = _determine_cutoff_indices(shape, kimax, kjmax, mode = "fft2")
        
        out = np.empty(shape = x.shape[:-2] + shape, dtype = x.dtype)
        out[...,:istop,:jstop] = x[...,:istop,:jstop] 
        out[...,-istop+1:,:jstop] = x[...,-istop+1:,:jstop] 
        out[...,:istop,-jstop+1:] = x[...,:istop,-jstop+1:] 
        out[...,-istop+1:,-jstop+1:] = x[...,-istop+1:,-jstop+1:] 
        return out
       
def rfft2_video(video, kimax = None, kjmax = None, overwrite_x = False, extra = {}):
    """A generator that performs rfft2 on a sequence of multi-frame data.
    
    Shape of the output depends on kimax and kjmax. It is (2*kimax+1, kjmax +1), 
    or same as the result of rfft2 if kimax and kjmax are not defined.
    
    Parameters
    ----------
    video : iterable
        An iterable of multi-frame data
    kimax : int, optional
        Max value of the wavenumber in vertical axis (i)
    kjmax : int, optional
        Max value of the wavenumber in horizontal axis (j)
    overwrite_x : bool, optional
        If input type is complex and fft library used is not numpy, fft can 
        be performed inplace to speed up computation.
    extra : dict
        Extra arguments passed to the underlying rfft2 cfunction. These arguments
        are library dependent. For pyffyw see the documentation on 
        additional arguments for finer FFT control.
        
    Returns
    -------
    video : iterator
        An iterator over FFT of the video.
    """
    for frames in video:
        yield tuple((rfft2_crop(_rfft2(frame, overwrite_x = overwrite_x, extra = extra),kimax, kjmax) for frame in frames))

def fft2_video(video, kimax = None, kjmax = None, overwrite_x = False, extra = {}):
    """A generator that performs fft2 on a sequence of multi-frame data.
    
    Shape of the output depends on kimax and kjmax. It is (2*kimax+1, 2*kjmax+1,), 
    or same as the result of fft2 if kimax and kjmax are not defined.
    
    Parameters
    ----------
    video : iterable
        An iterable of multi-frame data
    kimax : int, optional
        Max value of the wavenumber in vertical axis (i)
    kjmax : int, optional
        Max value of the wavenumber in horizontal axis (j)
    overwrite_x : bool, optional
        If input type is complex and fft library used is not numpy, fft can 
        be performed inplace to speed up computation.
    extra : dict
        Extra arguments passed to the underlying fft2 cfunction. These arguments
        are library dependent. For pyffyw see the documentation on 
        additional arguments for finer FFT control.
        
    Returns
    -------
    video : iterator
        An iterator over FFT of the video.
    """
    for frames in video:
        yield tuple((fft2_crop(_fft2(frame, overwrite_x = overwrite_x, extra = extra),kimax, kjmax) for frame in frames))

def _build_kf(window, kvec):
    window = np.asarray(window)
    kvec = np.asarray(kvec)
    if kvec.ndim not in (1,2) and kvec.shape[-1] != 2:
        raise ValueError("Invalid kvec")
    is_multi_kvec = True if kvec.ndim == 2 else False
    if is_multi_kvec:
        if window.ndim == 2:
            #add axis for broadcasting
            window = window[None,...]
    
    
    kfshape = kvec.shape[0:-1] + window.shape[-2:]
    kf = np.zeros(kfshape, CDTYPE)
    if is_multi_kvec:
        for i,k in enumerate(kvec):
            ki,kj = k
            kf[i,ki,kj] = 1.
    else:
        ki,kj = kvec
        kf[ki,kj] = 1.   
    kf = _fft2(kf, overwrite_x = True)  
    return kf

def _build_kernel(window, kf):
    kernel = (window*kf)
    return _fft2(kernel, overwrite_x = True)

def _convolve_fft(frame, kernel, kimax = None, kjmax = None):
    fout = _fft2(frame)
    fout = fout*kernel
    fout = fft2_crop(fout, kimax,kjmax)
    out = _ifft2(fout, overwrite_x = True)
    return out

def dls(frame, window, kvec, kimax = None, kjmax = None):
    """Performs DLS-like imaging. You define the desired k-vector and the desired
    illumination shape (the window function). This function computes the Fourier 
    transform, performs phase/amplitude modulation in the Fourier space and 
    recontructs the image in real space. The phase/amplitude modulation is 
    such that it filters only the defined frequency components with a spatial
    resolution determined by the window function.
    """
    kf = _build_kf(window, kvec)
    kernel = _build_kernel(window, kf)
    out = _convolve_fft(frame, kernel, kimax = kimax, kjmax = kjmax)
    #out *= kf
    return out
          
def dls_video(video, window, kvec, kimax = None, kjmax = None):
    """Same as dls, but for video objects"""
    kf = _build_kf(window, kvec)
    kernel = _build_kernel(window, kf)
    for frames in video:
        yield tuple((_convolve_fft(frame, kernel, kimax = kimax, kjmax = kjmax) for frame in frames))
          
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
            
@deprecated("This function was renamed to rfft2_crop and it will be removed in the future")
def rfft2(*args,**kwargs):
    return rfft2_video(*args,**kwargs)

@deprecated("This function was renamed to fft2_crop and it will be removed in the future")
def fft2(*args,**kwargs):
    return fft2_video(*args,**kwargs)
        
# if __name__ == "__main__":
#     import cddm.conf

#     from cddm.video import random_video, show_diff, show_video, show_fft, play
#     video = random_video(dual = True, shape = (512,256))
#     #video = show_video(video,0)
#     #video = show_diff(video)
#     video = show_fft(video,0)
#     #video = rfft2(video,63,63)
    
    
#     for frames in play(video, fps = 20):
#         pass
