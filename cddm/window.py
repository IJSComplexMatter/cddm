"""
Windowing functions. 

These windowing functions can be used to generate FFT window functions.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from cddm.conf import FDTYPE

def _r(shape, scale = 1.):
    """Returns radius array of a given shape."""
    ny,nx = [l/2 for l in shape]
    ay, ax = [np.arange(-l / 2. + .5, l / 2. + .5) for l in shape]
    xx, yy = np.meshgrid(ax, ay, indexing = "xy")
    r = ((xx/(nx*scale))**2 + (yy/(ny*scale))**2) ** 0.5    
    return r
        

def blackman(shape, out = None):
    """Returns a blacman window of a given shape.
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    window : ndarray
        A Blackman window
    """
    r = _r(shape)  
    if out is None:
        out = np.ones(shape, FDTYPE)
    out[...] = 0.42 + 0.5*np.cos(1*np.pi*r)+0.08*np.cos(2*np.pi*r)
    mask = (r>= 1.)
    out[mask] = 0.
    return out

def gaussian(shape, sigma, out = None):
    """Gaussian  window function.
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window
    sigma : float
        Waist of the gaussian in units of total width of the frame.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    window : ndarray
        Gaussian beam window
    """
    if sigma <= 0.:
        raise ValueError("Wrong sigma value")
    r = _r(shape, sigma* (2**0.5))
    if out is None:
        out = np.empty(shape, FDTYPE)
    return np.exp(-r**2, out = out)

def tukey(shape, alpha, out = None):
    """Returns a tukey window function. 
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window
    alpha : float
        Smoothnes parameter - defines smoothness of the edge of the tukey
        (should be between 0. and 1. When set to zero, it 
        becomes an aperture filter. When set to 1 it becomes a Hann window.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    window : ndarray
        Tukey window
    """
    r = _r(shape)
    return _tukey(r,alpha, out = out) 

def hann(shape, out = None):
    """Returns a hann window function. 
    
    Parameters
    ----------
    shape : (int,int)
        A shape of the 2D window
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    window : ndarray
        Hann window
    """
    r = _r(shape)
    return _tukey(r, 1., out = out) 

def _tukey(r,alpha = 0.1, rmax = 1., out =  None):
    if alpha> 1 or alpha < 0 :
        raise ValueError("Wrong alpha value")
    if out is None:
        out = np.ones(r.shape, FDTYPE)
    else:
        out[...] = 1.
    r = np.asarray(r, FDTYPE)
    alpha = alpha * rmax
    mask = r > rmax -alpha
    if alpha > 0.:
        tmp = 1/2*(1-np.cos(np.pi*(r-rmax)/alpha))
        out[mask] = tmp[mask]
    mask = (r>= rmax)
    out[mask] = 0.
    return out  

def plot_windows(shape = (256,256), alpha = 0.5, sigma = 0.4):
    """Plots all windows with a given shape, alpha (for tukey), sigma (for gaussian)
    values"""
    import matplotlib.pyplot as plt

    plt.subplot(221)
    im = plt.imshow(blackman(shape), vmin = 0, vmax = 1.)
    plt.colorbar(im)
    plt.xticks([])
    plt.title(r"blackman")
    plt.subplot(222)
    im = plt.imshow(hann(shape), vmin = 0, vmax = 1.)
    plt.xticks([])
    plt.colorbar(im)
    plt.title(r"hann")
    plt.subplot(223)
    im = plt.imshow(tukey(shape,alpha), vmin = 0, vmax = 1.)
    plt.colorbar(im)
    plt.title(r"tukey $\alpha = {:0.2}$".format(alpha))
    plt.subplot(224)
    im = plt.imshow(gaussian(shape,sigma), vmin = 0, vmax = 1.)
    plt.colorbar(im)
    plt.title(r"gaussian $\sigma = {:0.2}$".format(sigma))  
  
#
#if __name__  == "__main__":
#    plot_windows()
#
#    
