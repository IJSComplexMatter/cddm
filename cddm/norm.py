"""Normalization helper functions"""

from cddm._core_nb import weighted_sum, weight_from_g, weight_from_d, sigma2
import cddm._core_nb as _nb 
import numpy as np

def scale_factor(variance, mask = None):
    """Computes the normalization scaling factor from the variance data.
    
    You can divide the computed correlation data with this factor to normalize
    data between (0,1) for correlation mode, or (0,2) for difference mode.
    
    Parameters
    ----------
    variance : (ndarray, ndarray) or ndarray
        A variance data (as returned from :func:`.stats`)
    mask : ndarray
        A boolean mask array, if computation was performed on masked data,
        this applys mask to the variance data.
        
    Returns
    -------
    scale : ndarray
        A scaling factor for normalization
    """
    if variance is None:
        raise ValueError("You must provide variance data for normalization")
    try:
        v1, v2 = variance
        scale = (np.asarray(v1) + np.asarray(v2))/2
    except:
        scale = np.asarray(variance)  
    if mask is None:
        return scale
    else:
        return scale[mask]

def noise_delta(variance, mask = None):
    """Computes the scalled noise difference from the variance data.
    
    This is the delta parameter for weighted normalization.
    
    Parameters
    ----------
    variance : (ndarray, ndarray) 
        A variance data (as returned from :func:`.stats`)
    mask : ndarray
        A boolean mask array, if computation was performed on masked data,
        this applys mask to the variance data.
        
    Returns
    -------
    delta : ndarray
        Scalled delta value.
    """
    try:
        v1, v2 = variance
        delta = (np.asarray(v2) - np.asarray(v1))
        scale = (np.asarray(v1) + np.asarray(v2))
        delta /= scale
    except:
        #single data has delta = 0.
        scale = np.asarray(variance) 
        delta = np.zeros_like(scale)
    if mask is None:
        return delta
    else:
        return delta[mask]    

def weight_from_data(corr, scale_factor = 1., mode = "corr", pre_filter = True, delta = 0., out = None):
    """Computes weighting function for weighted normalization.
    
    Parameters
    ----------
    corr : ndarray
        Correlation (or difference) data
    scale_factor : ndarray
        Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
        corr data must be computed with scale = True option.
    mode : str
        Representation mode of the data, either 'corr' (default) or 'diff'
    pre_filter : bool
        Whether to perform denoising and filtering. If set to False, user has 
        to perform data filtering.
    delta : ndarray, optional
        The delta parameter, as returned by :func:`core.`
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
            corr = _nb.denoise(corr, out = out)
            corr = _nb.decreasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None], out = corr)
            corr = _nb.denoise(corr, out = corr)
            
        g = np.divide(corr,scale_factor[...,None], out = out)
         
        return weight_from_g(g,delta, out = out)
    
    elif mode == "diff":
        if pre_filter == True:
            corr = _nb.denoise(corr, out = out)
            corr = _nb.increasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2, out = corr)
            corr = _nb.denoise(corr, out = corr)
        d = np.divide(corr,scale_factor[...,None], out = out)
        return weight_from_d(d,delta, out = out)
    else:
        raise ValueError("Wrong mode.")
        
__all__ = ["weight_from_data","weighted_sum","scale_factor", "noise_delta",
           "weight_from_g", "weight_from_d","sigma2"]
