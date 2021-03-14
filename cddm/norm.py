"""Normalization helper functions"""

from __future__ import absolute_import, print_function, division


import numpy as np
from cddm.conf import CDTYPE, FDTYPE
from cddm.print_tools import print1,print2, enable_prints, disable_prints

from cddm._core_nb import _normalize_cdiff_1,_normalize_cdiff_3,\
  _normalize_ccorr_0,_normalize_ccorr_1,_normalize_ccorr_2,_normalize_ccorr_2b,_normalize_ccorr_3,_normalize_ccorr_3b

from cddm._core_nb import weighted_sum, weight_from_g, weight_from_d, sigma_weighted, sigma_prime_weighted,\
       weight_prime_from_g, weight_prime_from_d

import cddm._core_nb as _nb 
import cddm.avg as _avg

#NORM_BASELINE = 0
#"""baseline normalization flag"""

NORM_NONE = 0
"""none normalization flag"""

NORM_BASELINE = 0

# NORM_COMPENSATED = 1
# """compensated normalization (cross-diff) flag"""

# NORM_SUBTRACTED = 2
# """background subtraction normalization flag"""

# NORM_WEIGHTED = 4
# """weighted normalization flag"""


NORM_STANDARD = 1
"""standard normalization flag"""

NORM_STRUCTURED= 2
"""structured normalization flag"""

NORM_WEIGHTED = NORM_STANDARD | NORM_STRUCTURED
"""weighted normalization flag"""

NORM_SUBTRACTED = 4
"""background subtraction flag"""

NORM_COMPENSATED =8 # NORM_SUBTRACTED | 8
"""compensated normalization flag"""

import sys
_thismodule = sys.modules[__name__]

def norm_from_string(value):
    """Converts norm string to flags integer.
    
    Parameters
    ----------
    value : str 
        A string or combination of strings that describe the normalization. 
        Any of normalization modes 'baseline', 'subtracted', or 'compensated',
        You can mix these with calculation modes 'structured', 'standard', or 
        'weighted' by joining the normalization mode with the calculation
        mode e.g. "subtracted|structured"
        
    Returns
    -------
    norm : int
        Normalization flags.
    
    Examples
    --------
    >>> norm_from_string("structured")
    2
    
    """
    values = value.split("|")
    norm = 0
    for value in values:
        name = "NORM_" + value.strip().upper()
        try:
            norm = norm | getattr(_thismodule,name)
        except AttributeError:
            raise ValueError("Invalid norm string '{}'".format(value))
    if norm % 4 == 0:
        norm  = norm | NORM_STANDARD
    return norm

def norm_flags(structured = False, subtracted = False, weighted = False, compensated = False):
    """Return normalization flags from the parameters.
    
    Parameters
    ----------
    structured : bool
        Whether to set the STRUCTURED normalization flag.
    subtracted : bool
        Whether to set SUBTRACTED normalization flag.
    weighted : bool
        Whether to set WEIGHTED normalization flags.
    compensated : bool
        Whether to set COMPENSATED normalization flag.
    
    Returns
    -------
    norm : int
        Normalization flags.
        
    Examples
    --------
    >>> norm_flags(structured = True)
    2
    >>> norm_flags(compensated = True)
    9
    """
    standard = True if structured == False and weighted == False else False
    norm = NORM_NONE
    if standard == True :
        norm = norm | NORM_STANDARD
    if structured == True:
        norm = norm | NORM_STRUCTURED
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
    
def noise_level(window, intensity):
    """Computes the camera noise level spectrum in FFT space.
    
    This can be used to build noise and delta parameters for data error
    estimations.
    
    Parameters
    ----------
    window : ndarray
        Window function used in the analysis. Set this to np.ones, if no window
        is used. This is used to determine frame size
    intensity : float
        Mean image intensity.

    Returns
    -------
    noise : float
        Expected image noise level.
        
    Examples
    --------
    
    >>> window = np.ones(shape = (512,512))
    >>> noise_level(window, intensity = 200)
    52428800.0
    
    If you multiplied the image with a constant, e.g. 10 do
    
    >>> noise_level(window*10, intensity = 200)
    5242880000.0
    
    Noise level is 100 times larger in this case and is not the same as
    
    >>> noise_level(window, intensity = 2000)
    524288000.0
    """

    return (np.abs(window)**2).sum() * intensity
     
def noise_mean(corr, scale_factor = 1., mode = "corr"):
    """Computes the scalled mean noise from the  correlation data estimator.
    
    This is the delta parameter for weighted normalization.
    
    Parameters
    ----------
    corr: (ndarray,) 
        Correlation function (or difference function) model.
    scale_factor : ndarray
        Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
        corr data must be computed with scale = True option.
        
    Returns
    -------
    delta : ndarray
        Scalled delta value.
    """    
    scale_factor = np.asarray(scale_factor)
    g = np.divide(corr,scale_factor[...,None])
    if mode == "corr":
        noise = np.clip(1 - g[...,0],0,1) 
    elif mode == "diff":
        noise = np.clip(g[...,0]/2,0,1) 
    else:
        raise ValueError("Wrong mode.")
    return noise

def noise_delta(variance, mask = None, scale = True):
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
        scale = (np.asarray(v1) + np.asarray(v2)) if scale == True else 2.
        delta /= scale
    except:
        #single data has delta = 0.
        scale = np.asarray(variance) 
        delta = np.zeros_like(scale)
    if mask is None:
        return delta
    else:
        return delta[mask]   
    

def weight_from_data(corr, delta = 0., scale_factor = 1., mode = "corr", pre_filter = True):
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

        
    Returns
    -------
    out : ndarray
        Weight data for weighted sum calculation.
    """
    scale_factor = np.asarray(scale_factor, FDTYPE)
    delta = np.asarray(delta, FDTYPE)
    if mode == "corr":
        #make sure it is decreasing and clipped between 0 and 1
        if pre_filter == True:
            corr = _avg.denoise(corr)
            corr = _avg.decreasing(corr)
            corr = np.clip(corr,0.,scale_factor[...,None])
            corr = _avg.denoise(corr)
            
        g = np.divide(corr,scale_factor[...,None])
        
        return weight_from_g(g, delta[...,None])
    
    elif mode == "diff":
        if pre_filter == True:
            corr = _avg.denoise(corr)
            corr = _avg.increasing(corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2)
            corr = _avg.denoise(corr)
        
        d = np.divide(corr,scale_factor[...,None])

        return weight_from_d(d, delta[...,None])
    else:
        raise ValueError("Wrong mode.")

def weight_prime_from_data(corr, bg1, bg2, delta = 0., scale_factor = 1., mode = "corr", pre_filter = True):
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

        
    Returns
    -------
    out : ndarray
        Weight data for weighted sum calculation.
    """
    scale_factor = np.asarray(scale_factor, FDTYPE)
    delta = np.asarray(delta)
    bg1, bg2 = np.asarray(bg1), np.asarray(bg2)
    if mode == "corr":
        #make sure it is decreasing and clipped between 0 and 1
        if pre_filter == True:
            corr = _avg.denoise(corr)
            corr = _avg.decreasing(corr)
            corr = np.clip(corr,0.,scale_factor[...,None])
            corr = _avg.denoise(corr)
            
        g = np.divide(corr,scale_factor[...,None])
        
        return weight_prime_from_g(g, delta[...,None], bg1[...,None], bg2[...,None])
    
    elif mode == "diff":
        if pre_filter == True:
            corr = _avg.denoise(corr)
            corr = _avg.increasing(corr)
            corr = np.clip(corr,0.,scale_factor[...,None]*2)
            corr = _avg.denoise(corr)
        
        d = np.divide(corr,scale_factor[...,None])

        return weight_prime_from_d(d, delta[...,None], bg1[...,None],  bg2[...,None])
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
            default_norm = NORM_STANDARD
            available_norm = NORM_STANDARD
        else:
            default_norm = NORM_WEIGHTED
            available_norm = NORM_WEIGHTED
    else:
        if sq is None:
            default_norm = NORM_STANDARD | NORM_SUBTRACTED
            available_norm = NORM_STANDARD | NORM_COMPENSATED | NORM_SUBTRACTED
        else:
            default_norm = NORM_WEIGHTED | NORM_SUBTRACTED
            available_norm = NORM_WEIGHTED | NORM_COMPENSATED | NORM_SUBTRACTED
    if norm is None:
        return default_norm
    else:
        if norm not in (1,2,3,5,6,7,9,10,11,13,14,15):
            raise ValueError("Normalization mode must be one of (1,2,3,5,6,7,9,10,11,13,14,15)")
        if  (norm & available_norm) == norm:
            return norm
        else:
            raise ValueError("Normalization mode incompatible with input data.")

def _norm_from_cdiff_data(data, norm = None):
    """Determines normalization type from ccorr data"""
    available_norms = NORM_STRUCTURED, NORM_STRUCTURED | NORM_SUBTRACTED
    try:
        d, c, s1, s2 = data
    except:
        raise ValueError("Not a valid difference data")
    if (s1 is None or s2 is None):
        default_norm = available_norms[0]
    else:
        default_norm = available_norms[1]
    if norm is None:
        return default_norm
    else:
        if norm not in available_norms:
            raise ValueError("Normalization mode can be one of {}".format(available_norms))
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
        
        if mask is not None:
            bg1 = bg1[mask]
            bg2 = bg2[mask]
            
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
              scale = False, mask = None, weight = None, ret_weight = False, out = None):
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
    weight : ndarray, optional
        If you wish to specify your own weight for weighted normalization, you 
        must provide it here, otherwise it is computed from the data (default).
    ret_weight : bool, optional
        Whether to return weight (when calculating weighted normalization)
        
    out : ndarray, optional
        Output array
        
    Returns
    -------
    out : ndarray
        Normalized data.
    out, weight : ndarray, ndarray
        Normalized data and weight if 'ret_weight' was specified
    """
    print1("Normalizing...")
    #determine what kind of data is there ('diff' or 'corr')
    method = _method_from_data(data)
    scale = _inspect_scale(scale)
    mode = _inspect_mode(mode)
    
    if isinstance(norm,str):
        norm = norm_from_string(norm)
    
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
    
    bg1, bg2 = _inspect_background(background, norm, mask)
    
    if (norm & NORM_WEIGHTED == NORM_WEIGHTED):
        level = disable_prints()
        norm_comp = (norm & (NORM_SUBTRACTED | NORM_COMPENSATED)) | NORM_STRUCTURED
        comp_data = normalize(data, background, variance, norm = norm_comp,  mode = mode, 
              scale = scale, mask = mask)
        norm_base = (norm & (NORM_SUBTRACTED| NORM_COMPENSATED))| NORM_STANDARD
        base_data = normalize(data, background, variance, norm = norm_base,  mode = mode, 
              scale = scale, mask = mask)
        
        _multilevel = True if len(data[1].shape) > 1 else False
        
        x_avg,y_avg = _data_estimator(comp_data, size = 8, n = 3, multilevel = _multilevel)
        
        _scale_factor = 1. if scale == True else scale_factor(variance,mask)
        
        if weight is None:
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
        
            if norm & NORM_SUBTRACTED:
                delta = noise_delta(variance, mask, scale = scale)
                weight = weight_from_data(y_avg, delta, scale_factor = _scale_factor, mode = mode)
            else:
                bg1 = bg1 / (scale_factor(variance,mask))**0.5 if scale == True else bg1
                bg2 = bg2 / (scale_factor(variance,mask))**0.5  if scale == True else bg2
                delta = noise_delta(variance, mask, scale = scale)
                weight = weight_prime_from_data(y_avg, bg1, bg2, delta, scale_factor = _scale_factor, mode = mode)
            weight = _nb.log_interpolate(x_interp,x_avg, weight)
        
        enable_prints(level) 
        
        if ret_weight == True:
            return weighted_sum(comp_data, base_data,weight), weight
        else:
            return weighted_sum(comp_data, base_data,weight)
        
    count = data[1]
    
    #add dimensions for broadcasting
    bg1 = bg1[...,None]
    bg2 = bg2[...,None]
    
    if mask is not None:
        data = take_data(data, mask)
    
    #dimensions of correlation data (the first element of the data tuple)
    ndim = data[0].ndim
    
    #need to add dimensions to count for broadcasting
    #for 2D data, this is equivalent to count = count[...,None,None,:]
    ndiff = ndim - count.ndim
    for i in range(ndiff):
        count = np.expand_dims(count,-2)
        
    if norm == NORM_STANDARD:
        result = _normalize_ccorr_0(data[0], count, bg1, bg2, out = out)
        if mode == "diff":
            result = _corr2diff(result, variance, mask)
    
    elif norm == NORM_STRUCTURED:
        if method == "corr":
            offset = _variance2offset(variance, mask)
            result = _normalize_ccorr_1(data[0], count, bg1, bg2, data[2],  out = out)
            #m1 = data[3]
            #m2 = data[4] if data[4] is not None else m1
            #result = _normalize_ccorr_3b(data[0], count, data[2], m1, m2,  out = out)
            
            result += offset
            if mode == "diff":
                result = _corr2diff(result, variance, mask)
        
        else:
            d = bg2 - bg1
            result = _normalize_cdiff_1(data[0], count, d, out = out)
            if mode == "corr":
                result = _diff2corr(result, variance, mask)

    elif (norm == NORM_SUBTRACTED|NORM_STANDARD) or (norm == NORM_SUBTRACTED|NORM_STANDARD|NORM_COMPENSATED) :
        m1 = data[3]
        m2 = data[4] if data[4] is not None else m1
        result = _normalize_ccorr_2(data[0], count, bg1, bg2, m1,m2, out = out)
        if mode == "diff":
            result = _corr2diff(result, variance, mask)
            
    elif norm == NORM_STANDARD|NORM_COMPENSATED:
        m1 = data[3]
        m2 = data[4] if data[4] is not None else m1
        result = _normalize_ccorr_2b(data[0], count, m1,m2, out = out)
        if mode == "diff":
            result = _corr2diff(result, variance, mask)        

    elif (norm == NORM_STRUCTURED|NORM_SUBTRACTED)  or (norm == NORM_SUBTRACTED|NORM_STRUCTURED|NORM_COMPENSATED) :
        if method == "corr":
            offset = _variance2offset(variance, mask)
            m1 = data[3]
            m2 = data[4] if data[4] is not None else m1
            #result = _normalize_ccorr_2b(data[0], count, m1,m2, out = out)
            result = _normalize_ccorr_3(data[0], count, bg1, bg2, data[2], m1, m2,  out = out)
            result += offset
            if mode == "diff":
                result = _corr2diff(result, variance, mask)
        else:
            d = bg2 - bg1
            result = _normalize_cdiff_3(data[0], count, d, data[2], data[3],out = out)

            if mode == "corr":
                result = _diff2corr(result, variance, mask)
    elif norm == NORM_STRUCTURED|NORM_COMPENSATED:
        offset = _variance2offset(variance, mask)
        m1 = data[3]
        m2 = data[4] if data[4] is not None else m1
        result = _normalize_ccorr_3b(data[0], count, data[2], m1, m2,  out = out)
        result += offset
        if mode == "diff":
            result = _corr2diff(result, variance, mask)   
    else :
        raise ValueError("Unknown normalization mode {}".format(norm))                 
        
    if scale == True:
        result /= _scale_factor[...,None]
        
    if ret_weight == True:
        return result, weight
    else:
        return result


def _data_estimator(data, size = 8, n = 3, multilevel = False):
    from cddm.multitau import log_average, merge_multilevel
    from cddm.avg import denoise
    if multilevel == False:
        x,y = log_average(data,size)
    else:
        x,y = merge_multilevel(data)
    #in case we have nans, remove them before denoising, and return only valid data
    mask = np.isnan(y)
    mask = np.logical_not(np.all(mask,axis = tuple(range(mask.ndim-1))))
    return x[mask], denoise(y[...,mask], n = n)
        
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
           "weight_from_g", "weight_from_d","sigma_weighted","normalize", "take_data",
           "norm_flags","NORM_COMPENSATED","NORM_STANDARD","NORM_SUBTRACTED","NORM_WEIGHTED", "NORM_STRUCTURED"]
