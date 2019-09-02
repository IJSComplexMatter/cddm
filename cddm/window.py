#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windowing functions
"""
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
    waist : float
        Waist of the gaussian.
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    window : ndarray
        Gaussian beam window
    """
    r = _r(shape, sigma* (2**0.5))
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
        (should be between 0. and 1.; 0.05 by default). When set to zero, it 
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

if __name__  == "__main__":
    import matplotlib.pyplot as plt
    
    shape = (32,32) 
    #shape = (13,42) #any 2D shape is valid 
    
    plt.subplot(221)
    plt.imshow(blackman(shape))
    plt.title(r"blackman")
    plt.subplot(222)
    plt.imshow(hann(shape))
    plt.title(r"hann")
    plt.subplot(223)
    plt.imshow(tukey(shape,0.5))
    plt.title(r"tukey $\alpha = 0.5$")
    plt.subplot(224)
    plt.imshow(gaussian(shape,0.4))
    plt.title(r"gaussian $\sigma = 0.4$")  
    plt.show()
    
