"""Data averaging tools."""

from cddm._core_nb import log_interpolate, decreasing, increasing, median, \
        convolve, interpolate
import numpy as np

def denoise(x, n = 3):
    """Denoises data. A sequence of median and convolve filters.
    
    Parameters
    ----------
    x : ndarray
        Input array
    n : int
        Number of denoise steps (3 by default)
    out : ndarray, optional
        Output array
    
    Returns
    -------
    out : ndarray
        Denoised data.
    """
    for i in range(n):
        x = median(x)
        x = convolve(x)
    return x

__all__ = ["convolve","decreasing", "denoise", 
           "increasing", "interpolate","log_interpolate", "median"]