"""Normalization helper functions"""

from __future__ import absolute_import, print_function, division


import numpy as np
from cddm.conf import CDTYPE
from cddm.print_tools import print1,print2, enable_prints, disable_prints

from cddm._core_nb import _normalize_cdiff_1,_normalize_cdiff_3,\
  _normalize_ccorr_0,_normalize_ccorr_1,_normalize_ccorr_2,_normalize_ccorr_3

from cddm._core_nb import weighted_sum, weight_from_g, weight_from_d, sigma2
import cddm._core_nb as _nb 
import cddm.avg as _avg

NORM_BASELINE = 0
"""baseline normalization flag"""

NORM_COMPENSATED = 1
"""compensated normalization (cross-diff) flag"""

NORM_SUBTRACTED = 2
"""background subtraction normalization flag"""

NORM_WEIGHTED = 4
"""weighted normalization flag"""

def norm_flags(compensated = False, subtracted = False, weighted = False):
    """Return normalization flags from the parameters.
    
    Parameters
    ----------
    compensated : bool
        Whether to set COMPENSATED normalization flag
    subtracted : bool
        Whether to set SUBTRACTED normalization flag
    weighted : bool
        Whether to set WEIGHTED normalization flag
    
    Returns
    -------
    norm : int
        Normalization flagse
        
    Examples
    --------
    >>> norm_flags(True, False, True)
    5
    >>> norm_flags(True, True, False)
    3
    """
    norm = NORM_BASELINE
    if compensated == True:
        norm = norm | NORM_COMPENSATED
    if subtracted == True:
        norm = norm | NORM_SUBTRACTED
    if weighted == True:
        norm = norm | NORM_WEIGHTED
    return norm


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
            corr = _avg.denoise(corr, out = out)
            corr = _avg.decreasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None], out = corr)
            corr = _avg.denoise(corr, out = corr)
            
        g = np.divide(corr,scale_factor[...,None], out = out)
         
        return weight_from_g(g,delta, out = out)
    
    elif mode == "diff":
        if pre_filter == True:
            corr = _avg.denoise(corr, out = out)
            corr = _avg.increasing(corr, out = corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2, out = corr)
            corr = _avg.denoise(corr, out = corr)
        d = np.divide(corr,scale_factor[...,None], out = out)
        return weight_from_d(d,delta, out = out)
    else:
        raise ValueError("Wrong mode.")

def _norm_from_ccorr_data(data, norm = None):
    """Determines normalization type from ccorr data"""
    try:
        d, c, sq, s1, s2 = data
    except ValueError:
        raise ValueError("Not a valid correlation data")
        
    if s1 is None:
        if sq is None:
            default_norm = 0
            available_norm = 0
        else:
            default_norm = 4
            available_norm = 5
    else:
        if sq is None:
            default_norm = 2
            available_norm = 2
        else:
            default_norm = 6
            available_norm = 7
    if norm is None:
        return default_norm
    else:
        if norm not in range(8):
            raise ValueError("Normalization mode must be in range(8)")
        if  (norm & available_norm) == norm:
            return norm
        else:
            raise ValueError("Normalization mode incompatible with input data.")

def _norm_from_cdiff_data(data, norm = None):
    """Determines normalization type from ccorr data"""
    try:
        d, c, s1, s2 = data
    except:
        raise ValueError("Not a valid difference data")
    if (s1 is None or s2 is None):
        default_norm = 1
    else:
        default_norm = 3 
    if norm is None:
        return default_norm
    else:
        if norm not in (1,3):
            raise ValueError("Normalization mode can be 1 or 3")
        if  (norm & default_norm) == norm:
            return norm
        else:
            raise ValueError("Normalization mode incompatible with input data.")

def _default_norm_from_data(data, method, norm = None):
    if method == "corr":
        norm = _norm_from_ccorr_data(data, norm)
    else:
        norm = _norm_from_cdiff_data(data, norm)
    return norm
    

def _diff2corr(data, variance, mask):
    offset = _variance2offset(variance, mask)        
    data *= (-0.5)
    data += offset
    return data
    
def _corr2diff(data, variance, mask):
    offset = _variance2offset(variance, mask) * 2
    data *= (-2)
    data += offset
    return data

def _variance2offset(variance, mask):
    #offset for norm == 2 type of normalization
    return scale_factor(variance, mask)[..., None]
    

def _inspect_background(background, norm, mask):
    if background is not None:
        try:
            bg1, bg2 = background
        except:
            bg1, bg2 = background, background
            
        bg1, bg2 = np.asarray(bg1,CDTYPE),np.asarray(bg2,CDTYPE)
        bg1 = bg1[...,None] #add dimension for broadcasting
        bg2 = bg2[...,None]
        
        if mask is not None:
            bg1 = bg1[mask,:]
            bg2 = bg2[mask,:]
    elif norm > 0:
        raise ValueError("You must provide background data for normalization.")
    else:
        bg1, bg2 = 0j,0j
    return bg1, bg2
        
        
def _method_from_data(data):
    try:
        if len(data) == 5:
            return "corr" 
        elif len(data) == 4:
            return "diff"
        else:
            1/0
    except:
        raise ValueError("Not a valid correlation data")
        
def _inspect_mode(mode):
    if mode in ("corr","diff"):
        return mode
    else:
        raise ValueError("Invalid mode")
    
def _inspect_scale(scale):
    try:
        return bool(scale)
    except:
        raise ValueError("Invalid scale")
        
def normalize(data, background = None, variance = None, norm = None,  mode = "corr", 
              scale = False, mask = None, out = None):
    """Normalizes correlation (difference) data. Data must be data as returned
    from ccorr or acorr functions. 
    
    Except forthe most basic normalization, background and variance data must be provided.
    Tou can use :func:`stats` to compute background and variance data.
    
    Parameters
    ----------
    data : tuple of ndarrays
        Input data, a length 4 (difference data) or length 5 tuple (correlation data)
    background : (ndarray, ndarray) or ndarray, optional
        Background (mean) of the frame(s) in k-space
    variance : (ndarray, ndarray) or ndarray, optional
        Variance of the frame(s) in k-space
    norm : int, optional
        Normalization type (0:baseline,1:compensation,2:bg subtract,
        3: compensation + bg subtract). Input data must support the chosen
        normalization, otherwise exception is raised. If not given it is chosen
        based on the input data.
    mode : str, optional
        Representation mode: either "corr" (default) for correlation function,
        or "diff" for image structure function (image difference).
    scale : bool, optional
        If specified, performs scaling so that data is scaled beteween 0 and 1.
        This works in connection with variance, which must be provided.
    mask : ndarray, optional
        An array of bools indicating which k-values should we select. If not 
        given, compute at every k-value.
    out : ndarray, optional
        Output array
        
    Returns
    -------
    out : ndarray
        Normalized data.
    """
    print1("Normalizing...")
    #determine what kind of data is there ('diff' or 'corr')
    method = _method_from_data(data)
    scale = _inspect_scale(scale)
    mode = _inspect_mode(mode)
    
    #determine default normalization if not specified by user
    norm = _default_norm_from_data(data, method, norm)

#    if len(data[1].shape) > 1 and norm & NORM_WEIGHTED:
#        #if multilevel data, subtracted or baseline is best, so force it here.
#        norm = norm & NORM_SUBTRACTED

    # scale factor for normalization 
    if scale == True:
        _scale_factor = scale_factor(variance,mask)
    
    print2("   * background : {}".format(background is not None))
    print2("   * variance   : {}".format(variance is not None))
    print2("   * norm       : {}".format(norm))
    print2("   * scale      : {}".format(scale))
    print2("   * mode       : {}".format(mode))
    print2("   * mask       : {}".format(mask is not None))
    
    if (norm & NORM_WEIGHTED):
        level = disable_prints()
        norm_comp = (norm & NORM_SUBTRACTED)|NORM_COMPENSATED
        comp_data = normalize(data, background, variance, norm = norm_comp,  mode = mode, 
              scale = scale, mask = mask)
        norm_base = norm & NORM_SUBTRACTED
        base_data = normalize(data, background, variance, norm = norm_base,  mode = mode, 
              scale = scale, mask = mask)
        
        _multilevel = True if len(data[1].shape) > 1 else False
        
        x_avg,y_avg = _data_estimator(comp_data, size = 8, n = 3, multilevel = _multilevel)
        
        _scale_factor = 1. if scale == True else scale_factor(variance,mask)
        
        if _multilevel:
            shape = np.array(comp_data.shape)
            shape[1:-1] = 1
            _x_interp = np.arange(comp_data.shape[-1])
            x_interp = np.empty(shape = shape, dtype = int)
            for x_level in x_interp:
                x_level[...] = _x_interp
                _x_interp *= 2
        else:
            x_interp = np.arange(data[1].shape[-1])
        
        weight = weight_from_data(y_avg, scale_factor = _scale_factor, mode = mode)
        weight = _nb.log_interpolate(x_interp,x_avg, weight)
        enable_prints(level) 
        return weighted_sum(comp_data, base_data,weight)
        
    count = data[1]
    
    if mask is not None:
        data = take_data(data, mask)
    
    #dimensions of correlation data (the first element of the data tuple)
    ndim = data[0].ndim
            
    bg1, bg2 = _inspect_background(background, norm, mask)
    
    #need to add dimensions to count for broadcasting
    #for 2D data, this is equivalent to count = count[...,None,None,:]
    ndiff = ndim - count.ndim
    for i in range(ndiff):
        count = np.expand_dims(count,-2)

    if norm == NORM_BASELINE:
        result = _normalize_ccorr_0(data[0], count, bg1, bg2, out = out)
        if mode == "diff":
            result = _corr2diff(result, variance, mask)
    
    elif norm == NORM_COMPENSATED:
        if method == "corr":
            offset = _variance2offset(variance, mask)
            result = _normalize_ccorr_1(data[0], count, bg1, bg2, data[2],  out = out)
            result += offset
            if mode == "diff":
                result = _corr2diff(result, variance, mask)
        
        else:
            d = bg2 - bg1
            result = _normalize_cdiff_1(data[0], count, d, out = out)
            if mode == "corr":
                result = _diff2corr(result, variance, mask)

    elif norm == NORM_SUBTRACTED:
        m1 = data[3]
        m2 = data[4] if data[4] is not None else m1
        result = _normalize_ccorr_2(data[0], count, bg1, bg2, m1,m2, out = out)
        if mode == "diff":
            result = _corr2diff(result, variance, mask)

    elif norm == NORM_COMPENSATED|NORM_SUBTRACTED:
        if method == "corr":
            offset = _variance2offset(variance, mask)
            m1 = data[3]
            m2 = data[4] if data[4] is not None else m1
            result = _normalize_ccorr_3(data[0], count, bg1, bg2, data[2], m1, m2,  out = out)
            result += offset
            if mode == "diff":
                result = _corr2diff(result, variance, mask)
        else:
            d = bg2 - bg1
            result = _normalize_cdiff_3(data[0], count, d, data[2], data[3],out = out)

            if mode == "corr":
                result = _diff2corr(result, variance, mask)
    if scale == True:
        result /= _scale_factor[...,None]
    return result


def _data_estimator(data, size = 8, n = 3, multilevel = False):
    from cddm.multitau import log_average, merge_multilevel
    from cddm.avg import denoise
    if multilevel == False:
        x,y = log_average(data,size)
    else:
        x,y = merge_multilevel(data)
    return x, denoise(y, n = n, out = y)
        
def take_data(data, mask):
    """Selects correlation(difference) data at given masked indices.
    
    Parameters
    ----------
    data : tuple of ndarrays
        Data tuple as returned by `ccorr` and `acorr` functions
    mask : ndarray
        A boolean frame mask array 
    
    Returns
    -------
    out : tuple
        Same data structure as input data, but with all arrays in data masked
        with the provided mask array.
    """
    def _mask(i,data):
        if i !=1:
            data = data[...,mask,:] if data is not None else None
        return data
        
    return tuple((_mask(i,d) for (i,d) in enumerate(data)))



__all__ = ["weight_from_data","weighted_sum","scale_factor", "noise_delta",
           "weight_from_g", "weight_from_d","sigma2","normalize", "take_data",
           "norm_flags","NORM_COMPENSATED","NORM_BASELINE","NORM_SUBTRACTED","NORM_WEIGHTED"]
