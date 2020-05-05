"""Data averaging tools."""

import numpy as np
from cddm._core_nb import log_interpolate, decreasing, increasing, median, \
        convolve, interpolate, weighted_sum, weight_from_g1, weight_from_d

def denoise(x, n = 3, out = None):
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
        out = median(x, out = out)
        data = convolve(out, out = out)
    return data

def weight_from_data(corr, scale_factor = 1., mode = "corr", pre_filter = True, out = None):
    """Computes weighting function for weighted normalization.
    
    Parameters
    ----------
    corr : ndarray
        Correlation (or difference) data
    scale_factor : ndarray
        Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
        corr data must be computed with scale = True option.
    mode : str
        Representation mode, either 'corr' (default) or 'diff'
    pre_filter : bool
        Whether to perform denoising and filtering. If set to False, user has 
        to perform data filtering.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    out : ndarray
        Weight data for weighted sum calculation.
    """
    scale_factor = np.asarray(scale_factor)
    if mode == "corr":
        #make sure it is decreasing and clipped between 0 and 1
        if pre_filter == True:
            corr = denoise(corr, out = out)
            corr = decreasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None], out = corr)
            corr = denoise(corr, out = corr)
            
        g1 = np.divide(corr,scale_factor[...,None], out = out)
         
        return weight_from_g1(g1, out = out)
    
    elif mode == "diff":
        if pre_filter == True:
            corr = denoise(corr, out = out)
            corr = increasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2, out = corr)
            corr = denoise(corr, out = corr)
        d = np.divide(corr,scale_factor[...,None], out = out)
        return weight_from_d(d, out = out)
    else:
        raise ValueError("Wrong mode.")


__all__ = ["convolve","decreasing", "denoise", 
           "increasing", "interpolate","log_interpolate", "median","weight_from_data","weighted_sum",
           "weight_from_g1", "weight_from_d"]