"""Frames and image conversion functions"""

import numpy as np

from cddm.conf import FDTYPE, CDTYPE
from cddm.fft import _fft2, _rfft2, _ifft2, _irfft2
from cddm.map import from_rfft2_data, as_rfft2_data
from cddm._core_nb import phase , _abs2

def convert_image_space(im, in_space = "image", out_space = "image", shape = None, overwrite_x = False):
    """Converts image space.
    
    This is a convenience function, it performs conversion between real
    and reciprocial space. 
    
    Parameters
    ----------
    im : ndarray
        Input grayscale image (2D array).
    in_space : str
        One of "image", "rfft2", "fft2", describing input image space.
    out_space : str
        One of "image", "rfft2", "fft2", describing output image space.
    
    Returns
    -------
    out : ndarray
        Convertet image
    """
    
    if out_space == in_space:
        return im
    elif in_space == "image":
        if out_space == "fft2":
            return _fft2(im, overwrite_x = overwrite_x)
        elif out_space == "rfft2":
            return _rfft2(im)
        else:
            raise ValueError(f"Unsupported output type {out_space}.")
    elif in_space == "fft2":
        if out_space == "image":
            return _ifft2(im, overwrite_x = overwrite_x)
        elif out_space == "rfft2":
            return as_rfft2_data(im)
        else:
            raise ValueError(f"Unsupported output type {out_space}.")
    elif in_space == "rfft2":
        if out_space == "image":
            return _irfft2(im)
        elif out_space == "fft2":
            return from_rfft2_data(im,shape = shape)
        else:
            raise ValueError(f"Unsupported output type {out_space}.") 
    else:
        raise ValueError(f"Unsupportedinput type {in_space}.")

def data_repr(data, mode  = "real"):
    """Returns a new real representation of the input complex array.
    
    Parameters
    ----------
    data : ndarray
        Input (complex) array
    mode : str
        One of "real", "imag", "abs" or "phase" describing the representation
        mode of the complex data. 
        
    Returns
    -------
    out : ndarray
        In case mode is in ("real", "imag") it returns a view of the array,
        else it returns an new instance of the array with the specified mode.
    """
    if mode  == "real":
        return data.real
    elif mode  == "imag":
        return data.imag
    elif mode == "abs":
        return np.abs(data)
    elif mode  == "abs2":
        return _abs2(data)
    elif mode  == "phase":
        return phase(data)
    else:
        raise ValueError(f"Unsupported data representation mode  {mode}")
            
def normalize_image(im, overwrite_x = False):
    """Normalizes input image to mean intensity"""
    fact = 1./im.mean()
    out = im if overwrite_x == True else None
    return np.multiply(im,fact, out = out)

def normalize_fft(im, overwrite_x = False):
    """Normalizes input image to mean intensity in fft space"""
    fact = 1./im[0,0]
    out = im if overwrite_x == True else None
    return np.multiply(im,fact, out = out)

def image_repr(im, repr_mode  = "real", bg = None, overwrite_x = False):
    """Returns a new real representation of the input complex array.
    
    Parameters
    ----------
    im : ndarray
        Input (complex) image
    repr_mode : str
        One of "real", "imag", "abs" or "phase" describing the representation
        mode of the complex data. 
    bg : ndarray, optional
        If set, it describes the background image that is subtracted from the input image.
        
    Returns
    -------
    out : ndarray
        In case mode is in ("real", "imag") and bg is None, it returns a view 
        of the array, else it returns an new instance of the array with  
        background subtracted.
    """
    if bg is None:
        return data_repr(im, mode  = repr_mode)
    else:
        out = im if overwrite_x == True else None
        return data_repr(np.subtract(im, bg, out = out), mode  = repr_mode)

def frames_diff(frames, repr_mode  = "real", bg = None, overwrite_x = False):
    """Returns frame difference of the input (complex) frames.
    
    data_repr(im2 - im1) 
    
    Parameters
    ----------
    frames : (ndarray, ndarray)
        Input (complex) dual frame data.
    repr_mode : str
        One of "real", "imag", "abs" or "phase" describing the representation
        mode of the complex data. 
    bg : (ndarray,ndarray), optional
        If set, it describes the background image that is subtracted from the input image.
        
    Returns
    -------
    out : ndarray
        Background subtracted image difference.
    """
    im1, im2 = frames
    if bg is None:
        bg1, bg2 = None, None
    else:
        bg1, bg2 = bg
    if bg1 is not None:
        out = im1 if overwrite_x == True else None
        im1 = np.subtract(im1,bg1,out = out)
    if bg2 is not None:
        out = im1 if overwrite_x == True else None
        im2 = np.subtract(im2,bg2,out = out)
    out = im1 if overwrite_x == True else None
    return data_repr(np.subtract(im2,im1,out = out), mode  = repr_mode)

def frames_corr(frames, repr_mode  = "real", bg = None, autoscale = False, overwrite_x = False):
    """Returns a complex conjugate product of the input (complex) frames.
    
    data_repr(np.abs(im1) * im2) 
    
    Parameters
    ----------
    frames : (ndarray, ndarray)
        Input (complex) dual frame data.
    repr_mode : str
        One of "real", "imag", "abs" or "phase" describing the representation
        mode of the complex data. 
    bg : (ndarray,ndarray), optional
        If set, it describes the background image that is subtracted from the input image.
        
    Returns
    -------
    out : ndarray
        Background subtracted image difference.
    """
    im1, im2 = frames
    if bg is None:
        bg1, bg2 = None, None
    else:
        bg1, bg2 = bg
    if bg1 is not None:
        out = im1 if overwrite_x == True else None
        im1 = np.subtract(im1,bg1,out = out)
    if bg2 is not None:
        out = im1 if overwrite_x == True else None
        im2 = np.subtract(im2,bg2,out = out)
    return data_repr(np.conj(im2)*im1, mode  = repr_mode)

def calculate_bg(im, bg = None, count = 1, overwrite_x = False):
    """Calculates new background from a given input image and previously calculated
    background.
    
    Parameters
    ----------
    im : ndarray
        Next image
    bg : ndarray
        Previously calculated background
    count : int
        How many images have been taken already.
        
    Returns
    -------
    bg : ndarray
        Returns a modified background. This is the same object as input `bg`, 
        except if input `bg` was None.
    """
    if bg is None:
        bg = im.copy()
    else:
        if count > 0:
            fact = ((count-1.)/count)
            bg = np.multiply(bg,fact,bg)
            fact = 1./ count
            out = im if overwrite_x == True else None
            bg = np.add(bg,np.multiply(im,fact, out= out), out = bg)
    return bg

def asframes(frames, dtype = None):
    """Convert frames tu tuple of numpy arrays (frames)"""
    return tuple((np.asarray(frame, dtype) for frame in frames))

CONVERTER_PARAMETERS = ("in_space", "out_space", "repr_mode", "typ", "normalize", "auto_background", "navg")

class FramesConverter():
    """Main object for dual frames to image conversion.
    """
    def __init__(self, **kwargs):
        self.init(**kwargs)

    def init(self, in_space = "image", out_space = "image", repr_mode  = "real",
             typ = "cam1", normalize = False, auto_background = False, navg = 0):
        """Initializes or resets the converter
        
        Parameters
        ----------
        in_space : str
            One of "image", "rfft2", "fft2", describing input image space.
        out_space : str
            One of "image", "rfft2", "fft2", describing output image space.
        repr_mode : str
            One of "real", "imag", "abs2" or "phase" describing the representation
            mode of the (complex) data.  
        typ : str
            Frames converter type, one of "cam1", "cam2", "diff" or "corr"
        normalize : bool
            Whether to normalize each frame with mean intensity or not.
        auto_background : bool
            Whether to perform automatic background subtraction.
        navg : int
            How many frames to process in the averaging of the background. If set
            to zero, it take all acquired frames.
        """
        if in_space in ("image", "rfft2", "fft2"):
            self.in_space = in_space 
        else:
            raise ValueError(f"Invalid in_space {in_space}")
            
        self.out_space = out_space
        self.repr_mode  = repr_mode 
        self.typ = typ
        self.normalize = normalize
        self.auto_background = auto_background
        self.navg = navg
        self.count = 0
        self.bg = (None,None)
            
    def convert_frames(self,frames):
        """Converts input (dual) frames data to image.
        """
        frames = asframes(frames, FDTYPE) if self.in_space == "image" else asframes(frames, CDTYPE)
        self.count += 1
        if self.normalize == True:
            if self.in_space == "image":
                frames = (normalize_image(im) for im in frames)
            else:
                # if not image, then it must be fft data, so normalize fft
                frames = (normalize_fft(im) for im in frames)
        frames = (convert_image_space(im, in_space = self.in_space, out_space = self.out_space) for im in frames)
        #force caluclating frames from the iterator 
        frames = tuple(frames)
        #in case we 
        if self.auto_background == True:
            if self.navg < 1:
                self.bg = tuple((calculate_bg(im,bg = bg,count = self.count) for im,bg in zip(frames, self.bg)))
            else:
                self.bg = tuple((calculate_bg(im,bg = bg,count = self.navg) for im,bg in zip(frames, self.bg)))
              
        if self.typ in ("cam1" or 0):
            return image_repr(frames[0], bg = self.bg[0], repr_mode  = self.repr_mode )
        elif self.typ == ("cam1" or 1):
            return image_repr(frames[1], bg = self.bg[1], repr_mode  = self.repr_mode )          
        elif self.typ == ("diff" or 2):
            return frames_diff(frames, bg = self.bg, repr_mode  = self.repr_mode )
        elif self.typ == ("corr" or 3):
            return frames_corr(frames, bg = self.bg, repr_mode  = self.repr_mode )
        else:
            raise ValueError(f"Unknown type {type}")
            
    def __call__(self,value):
        return self.convert_frames(value)
