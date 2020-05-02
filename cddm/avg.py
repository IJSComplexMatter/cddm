"""Data averaging tools."""

import numpy as np
from cddm._core_nb import log_interpolate, decreasing, increasing, median, \
        convolve, interpolate, weighted_sum

WEIGHT_BASELINE = 0
"""Baseline weighting mode"""

WEIGHT_COMPENSATED = 1
"""Compensated weighting mode"""

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

def base_weight(corr, scale_factor = 1., mode = "corr", pre_filter = True, out = None):
    """Computes weighting function for baseline weighted normalization.
    
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
            
        #return (out/scale_factor[...,None] - 1)**2
        out = np.divide(corr, scale_factor[...,None], out = out)
        out -= 1
        out *= out 
        return out
    elif mode == "diff":
        if pre_filter == True:
            corr = denoise(corr, out = out)
            corr = increasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2, out = corr)
            corr = denoise(corr, out = corr)
        #return (0.5 * d/scale_factor[...,None])**2
        out = np.divide(corr,scale_factor[...,None], out = out)
        out *= 0.5
        out *= out
        return out
    else:
        raise ValueError("Wrong mode.")

def comp_weight(corr, scale_factor = 1., mode = "corr", pre_filter = True, out = None):
    """Computes weighting function for compensating weighted normalization.
    
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

        var2 = 2*(1 - g1)**2
        var1 = (1 + g1**2)
        return np.divide(var2,(var1 + var2), out = out)
    elif mode == "diff":
        if pre_filter == True:
            corr = denoise(corr, out = out)
            corr = increasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2, out = corr)
            corr = denoise(corr, out = corr)
        d = np.divide(corr,scale_factor[...,None], out = out)
        d /= 2.
        var2 = 2*(d)**2
        var1 = 1 + (d-1.)**2
        return np.divide(var2,(var1 + var2), out = out)
    else:
        raise ValueError("Wrong mode.")

def weight_from_data(x,xp,yp, scale_factor = 1., mode = "corr", norm = WEIGHT_BASELINE, pre_filter =True):
    """Computes weight at given x values from correlation data points (xp, yp).
    
    Parameters
    ----------
    x : ndarray
        x-values of the interpolated values
    xp : ndarray
        x-values of the correlation data
    yp : ndarray
        y-values of the correlation data, correlation data is over the last axis.
    scale_factor : ndarray
        Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
        corr data must be computed with scale = True option.
    mode : str
        Representation mode, either 'corr' (default) or 'diff'
    norm : int
        Weighting mode, 1 or odd for compensated, 0 or even for baseline. 
    pre_filter : bool
        Whether to perform denoising and filtering. If set to False, user has 
        to perform data filtering.
        
    Returns
    -------
    out : ndarray
        Weight data for weighted sum calculation.
    """
    if norm & WEIGHT_COMPENSATED:
        weight = comp_weight(yp, scale_factor = scale_factor, mode = mode, pre_filter = pre_filter)
    else:
        weight = base_weight(yp, scale_factor = scale_factor, mode = mode, pre_filter = pre_filter)
    return log_interpolate(x,xp,weight)


__all__ = ["base_weight", "comp_weight", "convolve","decreasing", "denoise", 
           "increasing", "interpolate","log_interpolate", "median","weight_from_data","weighted_sum"]