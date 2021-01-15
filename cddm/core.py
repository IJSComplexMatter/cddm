"""
Core functionality and main computation functions are defined here. There are
low level implementations.

* :func:`cross_correlate` 
* :func:`cross_correlate_fft`
* :func:`auto_correlate`
* :func:`auto_correlate_fft`
* :func:`cross_difference` 
* :func:`auto_difference`

along with functions for calculating tau-dependent mean signal and mean square
of the signal that are needed for normalization.

* :func:`cross_sum`
* :func:`cross_sum_fft`
* :func:`auto_sum`
* :func:`auto_sum_fft`

High-level functions include (in-memory calculation):

* :func:`ccorr` to calculate cross-correlation/difference functions
* :func:`acorr` to calculate auto-correlation/difference functions

For out-of-memory analysis use:

* :func:`iccorr` to calculate cross-correlation/difference functions
* :func:`iacorr` to calculate auto-correlation/difference functions

Finally, normalization of the results:
    
* :func:`normalize` to normalize the outputs of the above functions.

"""

from __future__ import absolute_import, print_function, division


import numpy as np
from cddm.conf import CDTYPE, FDTYPE, IDTYPE,I64DTYPE
from cddm.print_tools import print1,print2, print_frame_rate, enable_prints, disable_prints
import time
from functools import reduce 
from cddm.fft import _fft
from cddm._core_nb import _cross_corr_fft_regular, _cross_corr_fft, \
    _auto_corr_fft_regular,_auto_corr_fft,\
  _cross_corr_regular, _cross_corr, _cross_corr_vec, \
  _cross_diff_regular,_cross_diff,_cross_diff_vec, \
  _auto_corr_regular,_auto_corr, _auto_corr_vec, \
  _auto_diff_regular,_auto_diff, _auto_diff_vec,\
  _cross_sum_regular,_cross_sum,_cross_sum_vec,\
  _cross_sum_fft,_fill_ones,_cross_sum_rfft, abs2,\
  _add_count_cross,_add_count_auto

#imported for backward compatibility
from cddm.norm import NORM_STANDARD, NORM_SUBTRACTED, NORM_STRUCTURED, NORM_WEIGHTED, NORM_COMPENSATED
from cddm.norm import weight_from_data, weighted_sum, scale_factor, normalize, norm_flags, take_data

#--------------------------
#Core correlation functions

def _move_axis_and_align(f, axis, new_axis = -2, align = False):
    """Swaps axes of input data and aligns it by copying entire data."""
    f = np.asarray(f)
    if f.ndim > 1:    
        f = np.moveaxis(f,axis,new_axis)
        if align == True:
            f = f.copy()  
    return f

def _inspect_cross_arguments(f1,f2,t1,t2,axis,n, aout, dtype = CDTYPE):
    """Inspects and returns processed input arguments for cross_* functions"""
    if (t1 is None and t2 is not None) or (t2 is None and t1 is not None):
        raise ValueError("You must define both `t1` and `t2`")
    elif t1 is not None:
        t1 = np.asarray(t1, I64DTYPE)
        t2 = np.asarray(t2, I64DTYPE)
        
    axis = int(axis)
                
    f1 = np.asarray(f1, dtype)
    f2 = np.asarray(f2, dtype)
    
    if f1.dtype not in (FDTYPE, CDTYPE) and dtype is None:
        f1 = np.asarray(f1, FDTYPE)
    if f2.dtype not in (FDTYPE, CDTYPE) and dtype is None:
        f2 = np.asarray(f2, FDTYPE)
    
    ndim = f1.ndim

    if ndim == 0 or f1.shape != f2.shape:
        raise ValueError("Wrong shape of input arrays")

    if n is None:
        if aout is not None:
            if not isinstance(aout, np.ndarray):
                raise TypeError("aout must be a valid numpy array.")
            n = aout.shape[-1]
        else:
            n = _determine_full_length_cross(f1.shape,t1,t2,axis)
            
    if t1 is not None and t2 is not None:
        t1min,t2min = t1.min(), t2.min()
        t1 = t1 - min(t1min,t2min)
        t2 = t2 - min(t1min,t2min)
            
    return f1,f2,t1,t2,axis,n  


def _determine_full_length_auto(shape,t,axis):
    if t is None:
        #output size is input size
        n = shape[axis]
    else:
        #output size depends on the max and min values of t
        #determine appropriate length of the equivalent regular-spaced data
        tmin,tmax = t.min(), t.max()
        n =  1 + tmax -tmin 
    return n

def _determine_full_length_cross(shape,t1,t2,axis):
    if t1 is None and t2 is None:
        #output size is input size
        n = shape[axis]
    else:
        #output size depends on the max and min values of t
        #determine appropriate length of the equivalent regular-spaced data
        t1min,t1max = t1.min(), t1.max()
        t2min,t2max = t2.min(), t2.max()
        n =  1 + max(t2max-t1min, t1max - t2min)
    return n
    

def _inspect_auto_arguments(f,t,axis,n,aout, dtype = CDTYPE):
    """Inspects and returns processed input arguments for auto_* functions"""
    if t is not None:
        t = np.asarray(t, I64DTYPE)
        
    axis = int(axis)
                
    f = np.asarray(f, dtype)
    if f.dtype not in (FDTYPE, CDTYPE) and dtype is None:
        f = np.asarray(f, FDTYPE)
        
    ndim = f.ndim

    if ndim == 0:
        raise ValueError("Wrong shape of input arrays")

    if n is None:
        if aout is not None:
            if not isinstance(aout, np.ndarray):
                raise TypeError("aout must be a valid numpy array.")
            n = aout.shape[-1]
        else: 
            n = _determine_full_length_auto(f.shape,t,axis)
            
    if t is not None:
        tmin = t.min()
        t = t - tmin
        
    return f,t,axis,n  

def _transpose_data(data, axis = -2):
    return np.moveaxis(data,axis,-1)

def _default_out(out, data_shape, n, calc_axis, dtype = FDTYPE):
    shape = list(data_shape)
    shape[calc_axis] = n

    if out is None:
        out = np.zeros(shape, dtype)
    else:
        shape[calc_axis] = out.shape[-1]
        out = _transpose_data(out, calc_axis)
        if out.shape != tuple(shape):
            raise ValueError("Wrong output array shape")
        if out.dtype not in (FDTYPE,CDTYPE):
            raise ValueError("Wrong output dtype")
    return out

def cross_correlate_fft(f1,f2, t1 = None, t2 = None, axis = 0, n = None, aout = None):
    """Calculates cross-correlation function of two equal sized input arrays using FFT.
    
    For large arrays and large n, this is faster than correlate. The output of 
    this function is identical to the output of cross_correlate.
    
    See :func:`cross_correlate` for details.
    """
    #use None for "n" argument to determine length needed for FFT
    f1,f2,t1,t2,axis,n = _inspect_cross_arguments(f1,f2,t1,t2,axis,n,aout)
    #determine fft length
    length = _determine_full_length_cross(f1.shape,t1,t2,axis)
            
    #algorithm needs calculation to be done over the last axis.. so move it here
    f1 = np.moveaxis(f1,axis, -1)
    f2 = np.moveaxis(f2,axis, -1)
    
    out = _default_out(aout,f1.shape,n,-1)


    if t1 is None: 
        return _cross_corr_fft_regular(f1,f2,out, out)
    else:
        #random spaced data algorithm
        return _cross_corr_fft(f1,f2,t1,t2,length,out, out)

def auto_correlate_fft(f, t = None, axis = 0, n = None, aout = None):
    """Calculates auto-correlation function of input array using FFT.
    
    For large arrays and large n, this is faster than correlate. The output of 
    this function is identical to the output of auto_correlate.
    
    See :func:`auto_correlate` for details.
    """  
    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n,aout)
    #determine fft length
    length = _determine_full_length_auto(f.shape,t,axis)
                
    #algorithm needs calculation to be done over the last axis.. so move it here
    f = np.moveaxis(f,axis, -1)
    out = _default_out(aout,f.shape,n,-1)

    if t is None: 
        #regular spaced data algorithm
        return _auto_corr_fft_regular(f,out, out)
    else:
        #random spaced data algorithm
        return _auto_corr_fft(f,t,length,out, out)    

def _is_aligned(data, axis, align):
    try:
        return (((axis == -1 or axis == data.ndim-1) and data.data.contiguous == True) or align == True)    
    except:
        #python 2.7 just return False
        return False

def thread_frame_shape(shape, thread_divisor = None, force_2d = False):
    """Computes new frame shape for threaded computaton.
    
    Parameters
    ----------
    shape : tuple of ints
        Input frame shape
    thread_divisor : int
        An integer that divides the flattend frame shape. This number determines
        number of threads.
    force_2d : bool
        If 1d data, make it 2d regardless of thread_divisor value.
        
    Returns
    -------
    shape : tuple
        A length 2 shape
    """
    new_shape = shape
    if len(new_shape) == 1 and thread_divisor is None and force_2d == True:
        thread_divisor = 1
    if thread_divisor is not None:
        total = reduce((lambda x, y: x * y), new_shape)
        try:
            new_shape = (thread_divisor, total//int(thread_divisor))
        except ValueError:
            raise ValueError("Invalid `thread_divisor`")
        
        if total != new_shape[0] * new_shape[1]:
            raise ValueError("`thread_divisor` incompatible with input array's shape")
    return new_shape 
            
def reshape_input(f, axis = 0, thread_divisor = None, mask = None):
    """Reshapes input data, for faster threaded calculation
    
    Parameters
    ----------
    f : ndarray
        Input array
    axis : int
        Axis over which the computation is performed.
    thread_divisor : int
        An integer that divides the flattend frame shape. This number determines
        number of threads.
    mask : ndarray
        A boolean mask array. If provided, input data is masked first, then reshaped.
        This only works with axis = 0.
        
    Returns
    -------
    array, old_shape : ndarray, tuple
        Reshaped array and old frame shape tuple. Old frame shape is needed dor
        reshaping of output data with :func:`reshape_output`
    """
    
    f = np.asarray(f)
    axis = list(range(f.ndim))[axis]#make it a positive integer

    if mask is not None:
        if axis != 0:
            raise ValueError("Mask can only be applied when axis = 0")
        f = f[:,mask]   
        
    shape = list(f.shape)
    n = shape.pop(axis)
    force_2d = True if mask is None else False
    new_shape = list(thread_frame_shape(shape, thread_divisor, force_2d))
    new_shape.insert(axis, n)
    return f.reshape(new_shape), tuple(shape)

def reshape_frame(frame, shape, mask = None): 
    x = frame.reshape(shape)
    if mask is not None:
        out = np.empty(mask.shape, x.dtype)
        out[mask] = x
        out[np.logical_not(mask)] = np.nan
        return out
    else:
        return x
    
def reshape_output(data, shape = None, mask = None):
    """Reshapes output data as returned from ccorr,acorr functions
    to original frame shape data.
    
    
    If you used :func:`reshape_input` to reshape input data before call to `ccorr`
    or `acorr` functions. You must call this function on the output data to 
    reshape it back to original shape and unmasked input array. Missing data
    is filled with np.nan. 
    
    Parameters
    ----------
    data : tuple of ndarrays, or ndarray
        Data as returned by :func:`acorr` or :func:`ccorr` or a numpy array, or
        a numpy array, as returned by :func:`normalize`
    shape : tuple of ints
        shape of the input frame data.
    mask : ndarray, optional
        If provided, reconstruct reshaped data to original shape, prior to masking
        with mask.
        
    Returns
    -------
    out : ndarray
        Reshaped data, as if there was no prior call to reshape_input on the 
        input data of :func:`acorr` or :func:`ccorr` functions.

    """
    def _reshape(i,x):
        #if not count data
        if i != 1 and x is not None: 
            x = x.reshape(shape)
            if mask is not None:
                out = np.empty(x.shape[0:-2] + mask.shape + (x.shape[-1],), x.dtype)
                out[...,mask,:] = x
                out[...,np.logical_not(mask),:] = np.nan
                return out
            else:
                return x
        else:
            return x
    if shape is None:
        if mask is None:
            raise ValueError("Either `mask` or `shape` must be defined")
        shape = mask.shape
    if mask is None:
        shape = data[0].shape[0:-3] + tuple(shape) + (data[0].shape[-1],) 
    else:
        shape = data[0].shape[0:-2] + tuple(shape) + (data[0].shape[-1],) 
    #normalized data 
    if isinstance(data, np.ndarray):
        return _reshape(0,data) 
    #raw tuple data
    else: 
        return tuple((_reshape(i,x) for (i,x) in enumerate(data)))

def cross_correlate(f1,f2, t1 = None, t2 = None, axis = 0, n = None, align = False,aout = None):
    """Calculates cross-correlation function of two equal sized input arrays.

    This function performs 
    out[k] = sum_{i,j, where  k = abs(t1[i]-t2[j])} (real(f1[i]*conj(f2[j])) 
    
    Parameters
    ----------
    f1 : array-like
       First input array
    f2 : array-like
       Second input array
    t1 : array-like, optional
       First time sequence. If not given, regular-spaced data is assumed.
    t2 : array-like, optional
       Second time sequence. If not given, t1 time sequence is assumed. 
    axis : int, optional
       For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
       Determines the length of the output (max time delay - 1 by default). 
       Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
       Specifies whether data is aligned in memory first, before computation takes place.
       This may speed up computation in some cases (large n). Note that this requires
       a complete copy of the input arrays. 
    aout : ndarray, optional
       If provided, this must be zero-initiated output array to which data is
       added.
       
    Returns
    -------
    out : ndarray
        Computed cross-correlation.
       
    See also
    --------
    ccorr
    cddm.multitau.ccorr_multi
    """
    #check validity of arguments and make defaults
    f1,f2,t1,t2,axis,n = _inspect_cross_arguments(f1,f2,t1,t2,axis,n, aout)

    regular = False
    new_axis = -2 if f1.ndim > 1 else -1    

    if t1 is None and t2 is None:
        if _is_aligned(f1, axis, align):
            regular = True
            new_axis = -1
        else:
            #force irregular calculation (since it is faster in this cas)
            t1 = np.arange(f1.shape[axis])
            t2 = t1
        
    f1 = _move_axis_and_align(f1,axis,new_axis, align)
    f2 = _move_axis_and_align(f2,axis,new_axis, align)
    
    out = _default_out(aout,f1.shape,n,new_axis)
    
    if regular == True:
        out = _cross_corr_regular(f1,f2,out,out)
    else:
        if f1.ndim == 1:
            out = _cross_corr(f1,f2,t1,t2,out,out)
        else:
            out = _cross_corr_vec(f1,f2,t1,t2,out,out)
            
    return _transpose_data(out,new_axis)

def cross_difference(f1,f2, t1 = None, t2 = None, axis = 0, n = None, align = False, aout = None):
    """Calculates cross-difference (image structure) function of two equal 
    sized input arrays.
    
    This function performs 
    out[k] = sum_{i,j, where  k = abs(t1[i]-t2[j])} (abs(f2[j]-f1[i]))**2 
    
    Parameters
    ----------
    f1 : array-like
       First input array
    f2 : array-like
       Second input array
    t1 : array-like, optional
       First time sequence. If not given, regular-spaced data is assumed.
    t2 : array-like, optional
       Second time sequence. If not given, t1 time sequence is assumed. 
    axis : int, optional
       For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
       Determines the length of the output (max time delay - 1 by default). 
       Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
       Specifies whether data is aligned in memory first, before computation takes place.
       This may speed up computation in some cases (large n). Note that this requires
       a complete copy of the input arrays. 
    aout : ndarray, optional
       If provided, this must be zero-initiated output array to which data is
       added.
       
    Returns
    -------
    out : ndarray
        Computed cross-difference.
        
    See also
    --------
    ccorr
    cddm.multitau.ccorr_multi
    """
    #check validity of arguments and make defaults
    f1,f2,t1,t2,axis,n = _inspect_cross_arguments(f1,f2,t1,t2,axis,n,aout)

    regular = False
    new_axis = -2 if f1.ndim > 1 else -1    

    if t1 is None and t2 is None:
        if _is_aligned(f1, axis, align):
            regular = True
            new_axis = -1
        else:
            t1 = np.arange(f1.shape[axis])
            t2 = t1
        
    f1 = _move_axis_and_align(f1,axis,new_axis, align)
    f2 = _move_axis_and_align(f2,axis,new_axis, align)
        
    out = _default_out(aout,f1.shape,n,new_axis)
    
    if regular == True:
        out = _cross_diff_regular(f1,f2,out,out)
    else:
        if f1.ndim == 1:
            out = _cross_diff(f1,f2,t1,t2,out,out)
        else:
            out = _cross_diff_vec(f1,f2,t1,t2,out,out)
            
    return _transpose_data(out,new_axis)

def auto_correlate(f, t = None, axis = 0, n = None, align = False, aout = None):
    """Calculates auto-correlation function.
    
    This function performs 
    out[k] = sum_{i,j, where  k = j - i >= 0} (real(f[i]*conj(f[j])) 
    
    Parameters
    ----------
    f : array-like
       Input array
    t : array-like, optional
       Time sequence. If not given, regular-spaced data is assumed.
    axis : int, optional
       For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
       Determines the length of the output (max time delay - 1 by default). 
       Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
       Specifies whether data is aligned in memory first, before computation takes place.
       This may speed up computation in some cases (large n). Note that this requires
       a complete copy of the input arrays. 
    aout : ndarray, optional
       If provided, this must be zero-initiated output array to which data is
       added.
       
    Returns
    -------
    out : ndarray
        Computed auto-correlation.
        
    See also
    --------
    acorr
    cddm.multitau.acorr_multi
    """ 
    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n,aout)
    
    regular = False
    new_axis = -2 if f.ndim > 1 else -1    

    if t is None:
        if _is_aligned(f, axis, align):
            regular = True
            new_axis = -1
        else:
            t = np.arange(f.shape[axis])
        
    f = _move_axis_and_align(f,axis,new_axis, align)
    
    out = _default_out(aout,f.shape,n,new_axis) 
    
    if regular == True:
        out = _auto_corr_regular(f,out,out)
    else:
        #out = _auto_corr(f,t,out,out)
        if f.ndim > 1:
            out = _auto_corr_vec(f,t,out,out)
        else:
            #out = _auto_corr_vec(f[:,None],t,out[:,None],out[:,None])[:,0]
            out = _auto_corr(f,t,out,out)

    return _transpose_data(out,new_axis)


def auto_difference(f, t = None, axis = 0, n = None, align = False, aout = None):
    """Calculates auto-difference function.
    
    This function performs 
    out[k] = sum_{i,j, where  k = j - i >= 0} np.abs((f[i] - f[j]))**2 
    
    Parameters
    ----------
    f : array-like
       Input array
    t : array-like, optional
       Time sequence. If not given, regular-spaced data is assumed.
    axis : int, optional
       For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
       Determines the length of the output (max time delay - 1 by default). 
       Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
       Specifies whether data is aligned in memory first, before computation takes place.
       This may speed up computation in some cases (large n). Note that this requires
       a complete copy of the input arrays. 
    aout : ndarray, optional
       If provided, this must be zero-initiated output array to which data is
       added.
       
    Returns
    -------
    out : ndarray
        Computed auto-difference.
        
    See also
    --------
    acorr
    cddm.multitau.acorr_multi
    """   
    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n, aout)
    
    regular = False
    new_axis = -2 if f.ndim > 1 else -1   
    
    if t is None:
        if _is_aligned(f, axis, align):
            regular = True
            new_axis = -1
        else:
            t = np.arange(f.shape[axis])
        
    f = _move_axis_and_align(f,axis,new_axis, align)
    
    out = _default_out(aout,f.shape,n,new_axis) 
    if regular == True:
        out = _auto_diff_regular(f,out,out)
    else:
        if f.ndim > 1:
            out = _auto_diff_vec(f,t,out,out)
        else:
            out = _auto_diff(f,t,out,out)

    return _transpose_data(out,new_axis)

def _default_count(t1,t2, n, out):
    if out is None:
        if n is None:
            t1 = np.asarray(t1)
            t2 = np.asarray(t2)
            t1min,t1max = t1.min(), t1.max()
            t2min,t2max = t2.min(), t2.max()
            n =  1 + max(t2max-t1min, t1max - t2min)

        out = np.zeros((int(n),), IDTYPE)
    else:
        if out.dtype != IDTYPE or out.ndim < 1:
            raise ValueError("Input array is not integer dtype or has wrong dimensions.")
    return out

def cross_count(t1,t2 = None, n = None, aout = None):
    """Culculate number of occurences of possible time delays in cross analysis
    for a given set of time arrays.
    
    Parameters
    ----------
    t1 : array_like or int
       First time array. If it is a scalar, assume regular spaced data of length 
       specified by t1.
    t2 : array_like or None
       Second time array. If it is a scalar, assume regular spaced data of length 
       specified by t2. If not given, t1 data is taken.
    n : int, optional
       Determines the length of the output (max time delay - 1 by default). 
       Note that 'aout' parameter takes precedence over 'n'.
    aout : ndarray, optional
       If provided, this must be zero-initiated output array to which data is
       added. If defeined, this takes precedence over the 'n' parameter.
       
    Examples
    --------
    >>> cross_count(10,n=5)
    array([10, 18, 16, 14, 12])
    >>> cross_count([1,3,6],[0,2,6],n=5)
    array([1, 3, 0, 2, 1])
    """
    t1 = np.asarray(t1, I64DTYPE)
    if t1.ndim == 0:
        #then t1 is the length of regular spaced data, so make t1 
        t1 = np.arange(t1)
    if t2 is None:
        t2 = t1
    else:
        t2 = np.asarray(t2, I64DTYPE)
        if t2.ndim == 0:
            t2 = np.arange(t2)
    out = _default_count(t1,t2,n, aout)
    _add_count_cross(t1,t2,out)
    return out

def auto_count(t, n = None, aout = None):
    """Culculate number of occurences of possible time delays in auto analysis
    for a given time array.
    
    Parameters
    ----------
    t : array_like or int
        Time array. If it is a scalar, assume regular spaced data of length specified by 't'
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
        Note that 'aout' parameter takes precedence over 'n'.
    aout : ndarray, optional
        If provided, this must be zero-initiated output array to which data is
        added. If defeined, this takes precedence over the 'n' parameter.
       
    Examples
    --------
    >>> auto_count(10)
    array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
    >>> auto_count([0,2,4,5])
    array([4, 1, 2, 1, 1, 1])
    """
    t = np.asarray(t, I64DTYPE)
    if t.ndim == 0:
        #then t is the length of regular spaced data, so make t
        t = np.arange(t)   
    out = _default_count(t,t,n, aout)
    _add_count_auto(t,out)
    return out

#------------------------------------------------------
# delay-dependent data sum function - for normalizations 

def cross_sum(f, t = None, t_other = None, axis = 0, n = None, align = False, aout = None):
    """Calculates sum of array, useful for normalization of correlation data.
    
    This function performs:
    out[k] = sum_{i,j, where  k = abs(t[i]-t_other[j])} (f[i])

    Parameters
    ----------
    f : array-like
        Input array
    t1 : array-like, optional
        Time sequence of iput array. If not given, regular-spaced data is assumed.
    t2 : array-like, optional
        Time sequence of the other array. If not given, t1 time sequence is assumed. 
    axis : int, optional
        For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
        Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
        Specifies whether data is aligned in memory first, before computation takes place.
        This may speed up computation in some cases (large n). Note that this requires
        a complete copy of the input arrays. 
    aout : ndarray, optional
        If provided, this must be zero-initiated output array to which data is
        added.
       
    Returns
    -------
    out : ndarray
        Calculated sum.
    """
    
    f,f,t,t_other,axis,n = _inspect_cross_arguments(f,f,t,t_other,axis,n,aout, dtype = None)

    if t is None and t_other is None:
        regular = True
        new_axis = -1
    else:
        regular = False
        new_axis = -2 if f.ndim > 1 else -1

    #transpose input data so that calculation axis is either -1 or -2, optionally, copy data to make it aligned in memory    
    f = _move_axis_and_align(f,axis, new_axis, align)
    
    out = _default_out(aout,f.shape,n,new_axis, dtype = f.dtype) 
    
    if regular == True:
        out = _cross_sum_regular(f,out,out)
    else:
        if f.ndim == 1: 
            out = _cross_sum(f,t,t_other,out,out)
        else:   
            out = _cross_sum_vec(f,t,t_other,out,out)

    out = _transpose_data(out, new_axis)
    return out

def cross_sum_fft(f, t, t_other = None, axis = 0, n = None, aout = None):
    """Calculates sum of array, useful for normalization of correlation data.
    
    This function is defined for irregular-spaced data only.
    
    See :func:`cross_sum` for details.
    """

    f,f,t,t_other,axis,n = _inspect_cross_arguments(f,f,t,t_other,axis,n, aout,dtype = None)

    length = _determine_full_length_cross(f.shape,t,t_other,axis)
        
    #algorithm needs calculation to be done over the last axis.. so move it here
    f = np.moveaxis(f,axis, -1)
    
    out = _default_out(aout,f.shape,n,-1, dtype = f.dtype) 

    if t is None: 
        raise ValueError("Not implemented for regular spaced data! Use cross_sum instead.")
    else:        
        #we need conjugate version of FFT of zero-padded ones at t_other times
        y = np.zeros((length*2), CDTYPE)
        _fill_ones(t_other, y)
        y = _fft(y, overwrite_x = True)
        np.conj(y, out = y)
        
        if np.iscomplexobj(f):
            return _cross_sum_fft(f,y,t,length,out, out)
        else:
            return _cross_sum_rfft(f,y,t,length,out, out)


def auto_sum(f, t = None, axis = 0, n = None, align = False, aout = None):
    """Calculates sum of array, useful for normalization of autocorrelation data.
    
    This function performs:
    out[k] = sum_{i,j, where  k = abs(t[i]-t[j]), j >= i} (f[i]+f[j])/2.

    Parameters
    ----------
    f : array_like
        Input array
    t1 : array-like, optional
        Time sequence of iput array. If not given, regular-spaced data is assumed.
    t2 : array-like, optional
        Time sequence of the other array. If not given, t1 time sequence is assumed. 
    axis : int, optional
        For multi-dimensional arrays this defines computation axis (0 by default)
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
        Note that 'aout' parameter takes precedence over 'n'.
    align : bool, optional
        Specifies whether data is aligned in memory first, before computation takes place.
        This may speed up computation in some cases (large n). Note that this requires
        a complete copy of the input arrays. 
    aout : ndarray, optional
        If provided, this must be zero-initiated output array to which data is
        added.
       
    Returns
    -------
    out : ndarray
        Computed sum.
    """
    
    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n,aout,dtype = None)

    if t is None:
        regular = True
        new_axis = -1
    else:
        regular = False
        new_axis = -2 if f.ndim > 1 else -1

    #transpose input data so that calculation axis is either -1 or -2, optionally, copy data to make it aligned in memory    
    f = _move_axis_and_align(f,axis, new_axis, align)
    
    out = _default_out(aout,f.shape,n,new_axis, dtype = f.dtype) 
    out0 = np.zeros_like(out)
    
    if regular == True:
        out0 = _cross_sum_regular(f,out0,out0)
    else:
        if f.ndim == 1: 
            out0 = _cross_sum(f,t,t,out0,out0)
        else:   
            out0 = _cross_sum_vec(f,t,t,out0,out0)

    out0 = _transpose_data(out0, new_axis)
    out =_transpose_data(out, new_axis)
    #we used cross sum to calculate.. we counted all but first data twice
    out0[...,1:] /= 2.
    
    out += out0
    return out

def auto_sum_fft(f, t, axis = 0, n = None, aout = None):
    """Calculates sum of array, useful for normalization of correlation data.
    
    This function is defined for irregular-spaced data only.
    
    See :func:`auto_sum` for details.
    """

    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n,aout, dtype = None)
    
    length = _determine_full_length_auto(f.shape,t,axis)
        
    #algorithm needs calculation to be done over the last axis.. so move it here
    f = np.moveaxis(f,axis, -1)
    
    out = _default_out(aout,f.shape,n,-1, dtype = f.dtype) 
    out0 = np.zeros_like(out)
        
    if t is None: 
        raise ValueError("Not implemented for regular spaced data! Use auto_sum instead.")
    else:        
        #we need conjugate version of FFT of zero-padded ones at t_other times
        y = np.zeros((length*2), CDTYPE)
        _fill_ones(t, y)
        y = _fft(y, overwrite_x = True)
        np.conj(y, out = y)
        
        if np.iscomplexobj(f):
            out0 = _cross_sum_fft(f,y,t,length,out0, out0)
        else:
            out0 = _cross_sum_rfft(f,y,t,length,out0, out0)
        #we used cross sum to calculate.. we counted all but first data twice
        out0[...,1:] /= 2.
        out += out0
        return out


def subtract_background(data, axis=0, bg = None, return_bg = False, out = None):
    """Subtracts background frame from a given data array. 
    
    This function can be used to subtract user defined background data, or to
    compute and subtract background data.
    """
    print1("Subtracting background...")
    print2("   * axis : {}".format(axis))
    if bg is None:
       bg = np.mean(data, axis = axis)
    data = np.subtract(data,np.expand_dims(bg,axis),out)
    if return_bg == False:
        return data
    else:
        return data, bg

def stats(f1,f2 = None, axis = 0):
    """Computes statistical parameters for normalization of correlation data.
    
    Parameters
    ----------
    f1 : ndarray
        Fourier transform of the first video.
    f2 : ndarray, optional
        Second data set (for dual video)  
    axis : int, optional
        Axis over which to compute the statistics.
    Returns
    -------
    (f1mean, f2mean), (f1var, f2var): (ndarray, ndarray), (ndarray, ndarray)
        Computed mean and variance data of the input arrays.
    """
    print1("Computing stats...")
    print2("   * axis : {}".format(axis))
    f1 = np.asarray(f1)
    if f2 is None:
        return f1.mean(axis), f1.var(axis)
    else:
        f2 = np.asarray(f2)
        return (f1.mean(axis), f2.mean(axis)), (f1.var(axis), f2.var(axis))

#-------------------------------------
# main correlation functions

def _default_method(method, n):
    if method is None:
        method = "fft" if n is None else "corr"
    elif method not in ("corr","fft","diff"):
        raise ValueError("Unknown method")
    return method

def _default_norm(norm, method, cross = True):
    if method == "diff":
        supported = (NORM_STRUCTURED,NORM_STRUCTURED|NORM_SUBTRACTED) if cross else (NORM_STRUCTURED,)
    else:
        supported = (1,2,3,5,6,7)
    if norm is None:
        norm = supported[-1]
    if norm not in supported:
        raise ValueError("Norm {} not supported for method '{}'".format(norm,method))
    return norm

def acorr(f, t = None, fs = None,  n = None,  norm = None, 
           method = None, align = False, axis = 0, aout = None):  
    """Computes auto-correlation of the input signals of regular or irregular 
    time - spaced data.
    
    If data has ndim > 1, autocorrelation is performed over the axis defined by 
    the axis parameter. If 'aout' is specified the arrays must be zero-initiated.
    
    Parameters
    ----------
    f : array-like
        A complex ND array..
    t : array-like, optional
        Array of integers defining frame times of the data. If not provided, 
        regular time-spaced data is assumed.
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
        Note that 'aout' parameter takes precedence over 'n'
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3. Default to 3, except for 
        'diff' method where it default to 1.
    method : str, optional
        Either 'fft' , 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    align : bool, optional
        Whether to align data prior to calculation. Note that a complete copy of 
        the data takes place.
    axis : int, optional
        Axis over which to calculate.
    aout : a tuple of ndarrays, optional
        Tuple of output arrays. 
        For method =  'diff' : (corr, count, _, _) 
        for 'corr' and 'fft' : (corr, count, squaresum, sum, _)
       
    Returns
    -------
    (corr, count, squaresum, sum, _) : (ndarray, ndarray, ndarray, ndarray, NoneType)
        Computed correlation data for 'fft' and 'corr' methods,
        If norm = 3, these are all defined. For norm < 3, some may be NoneType.    
    (diff, count, _, _) : (ndarray, ndarray, NoneType, NoneType)
        Computed difference data for 'diff' method.
    """  
    t0 = time.time()

    method = _default_method(method, n)
    print1("Computing {}...".format(method))
    norm = _default_norm(norm, method, cross = False)
    
    correlate = False if method == "diff" else True
    
    if method == "diff":
        cor, count, _, _ = (None,)*4 if aout is None else aout 
    else:
        cor, count, sq, ds, _ = (None,)*5 if aout is None else aout 

    f,t,axis,n = _inspect_auto_arguments(f,t,axis,n, cor)
    
    nframes = f.shape[axis]

    regular = False
    new_axis = -2 if f.ndim > 1 else -1    

    if t is None:
        regular = True
        if _is_aligned(f, axis, align):
            new_axis = -1
            
    f = _move_axis_and_align(f,axis,new_axis, align)
    
    print2("   * axis   : {}".format(axis))
    print2("   * norm   : {}".format(norm))
    print2("   * n      : {}".format(n))
    print2("   * align  : {}".format(align))
    print2("   * method : {}".format(method))

    if method == "fft":
        cor = auto_correlate_fft(f,t, axis = new_axis, n = n, aout = cor)
    elif method == "corr":
        cor = auto_correlate(f,t, axis = new_axis, n = n, aout = cor)
    else:
        cor = auto_difference(f,t, axis = new_axis, n = n, aout = cor)
    if t is None:
        count = auto_count(np.arange(f.shape[new_axis]),n, aout = count)
    else:
        count = auto_count(t,n, aout = count)
    
    if method == "fft" and regular == False:
        _sum = auto_sum_fft
    else:
        _sum = auto_sum
        
    if (norm & NORM_STRUCTURED) and correlate:
        if fs is None:
            fs =  abs2(f)
        else:
            fs = _move_axis_and_align(fs,axis, new_axis, align)
    
        print2("... Computing asquare...")
        sq = _sum(2*fs,t,axis = new_axis, n = n, aout = sq) 
    
    if norm & NORM_SUBTRACTED and correlate:
        print2("... Computing amean...")
        
        ds = _sum(f,t = t,axis = new_axis, n = n, aout = ds)
        
    print_frame_rate(nframes, t0)
    if method == "diff":
        return cor, count, None, None
    else:
        return cor, count, sq, ds, None
 
def ccorr(f1,f2,t1 = None, t2 = None,  n = None, 
          norm = None,  method = None, 
          align = False, axis = 0, f1s = None, f2s = None, aout = None):  
    """Computes cross-correlation of the input signals of regular or irregular 
    time - spaced data.
    
    If data has ndim > 1, calculation is performed over the axis defined by 
    the axis parameter. If 'aout' is specified the arrays must be zero-initiated.
    
    Parameters
    ----------
    f1 : array-like
        A complex ND array of the first data.
    f2 : array-like
        A complex ND array of the second data.
    t1 : array-like, optional
        Array of integers defining frame times of the first data. If not provided, 
        regular time-spaced data is assumed.
    t2 : array-like, optional
        Array of integers defining frame times of the second data. If not provided, 
        regular time-spaced data is assumed.  
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
        Note that 'aout' parameter takes precedence over 'n'
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3 (default).
    method : str, optional
        Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    align : bool, optional
        Whether to align data prior to calculation. Note that a complete copy of 
        the data takes place.
    axis : int, optional
        Axis over which to calculate.
    f1s : array-like, optional
        First absolute square of the input data. For norm = NORM_COMPENSATED square of the 
        signal is analysed. If not given it is calculated on the fly.
    f2s : array-like, optional
        Second absolute square of the input data.
    aout : a tuple of ndarrays, optional
        Tuple of output arrays. 
        For method =  'diff' : (corr, count, sum1, sum2) 
        for 'corr' and 'fft' : (corr, count, squaresum, sum1, sum2)
       
    Returns
    -------
    (corr, count, squaresum, sum1, sum2) : (ndarray, ndarray, ndarray, ndarray, ndarray)
        Computed correlation data for 'fft' and 'corr' methods,
        If norm = 3, these are all defined. For norm < 3, some may be NoneType.    
    (diff, count, sum1, sum2) : (ndarray, ndarray, ndarray, ndarray)
        Computed difference data for 'diff' method.
       
    Examples
    --------
    
    Say we have two datasets f1 and f2. To compute cross-correlation of both 
    datasets :
        
    >>> f1, f2 = np.random.randn(24,4,6) + 0j, np.random.randn(24,4,6) + 0j
    
    >>> data = ccorr(f1, f2, n = 16)
    
    Now we can set the 'out' parameter, and the results of the next dataset
    are added to results of the first dataset:
    
    >>> data = ccorr(f1, f2,  aout = data)
    
    Note that the parameter 'n' = 64 is automatically determined here, based on the
    provided 'aout' arrays.  
    """
    t0 = time.time()
    
    method = _default_method(method, n)
    print1("Computing {}...".format(method))
    norm = _default_norm(norm, method, cross = True)
    
    correlate = False if method == "diff" else True
    
    if method == "diff":
        cor, count, ds1, ds2 = (None,)*4 if aout is None else aout 
    else:
        cor, count, sq, ds1, ds2 = (None,)*5 if aout is None else aout 

    f1,f2,t1,t2,axis,n = _inspect_cross_arguments(f1,f2,t1,t2,axis,n,cor)
    
    nframes = f1.shape[axis]
    
    regular = False
    new_axis = -2 if f1.ndim > 1 else -1    

    if t1 is None and t2 is None:
        regular = True
        if _is_aligned(f1, axis, align):
            new_axis = -1

    f1 = _move_axis_and_align(f1,axis,new_axis, align)
    f2 = _move_axis_and_align(f2,axis,new_axis, align)
    
    print2("   * axis   : {}".format(axis))
    print2("   * norm   : {}".format(norm))
    print2("   * n      : {}".format(n))
    print2("   * align  : {}".format(align))
    print2("   * method : {}".format(method))
 
    if norm != 0:
        print2("... correlate")

    if method == "fft":
        cor = cross_correlate_fft(f1,f2,t1,t2, axis = new_axis, n = n, aout = cor)
    elif method == "corr":
        cor = cross_correlate(f1,f2,t1,t2, axis = new_axis, n = n, aout = cor)
    else:
        cor = cross_difference(f1,f2,t1,t2, axis = new_axis, n = n, aout = cor)
    if t1 is None:
        t = np.arange(f1.shape[new_axis])
        count = cross_count(t,t,n, aout = count)
    else:
        count = cross_count(t1,t2,n, aout = count)
    
    if method == "fft" and regular == False:
        _sum = cross_sum_fft
    else:
        _sum = cross_sum

        
    if (norm & NORM_STRUCTURED) and correlate:
        print2("... tau sum square")
        if f1s is None:
            f1s =  abs2(f1)
        else:
            f1s = _move_axis_and_align(f1s,axis, new_axis, align)
            
        if f2s is None:
            f2s = abs2(f2)
        else:
            f2s = _move_axis_and_align(f2s,axis, new_axis, align)
            
        sq = _sum(f1s,t1,t_other = t2,axis = new_axis, n = n, aout = sq) 
        sq = _sum(f2s,t2,t_other = t1,axis = new_axis, n = n ,aout = sq) 
    
    if norm & NORM_SUBTRACTED:
        print2("... tau sum signal")
        
        ds1 = _sum(f1,t = t1,t_other = t2,axis = new_axis, n = n, aout = ds1)
        ds2 = _sum(f2,t = t2,t_other = t1,axis = new_axis, n = n, aout = ds2)
    
    print_frame_rate(nframes, t0)
    
    if method == "diff":
        return cor, count, ds1, ds2
    else:
        return cor, count, sq, ds1, ds2
    

def iccorr(data, t1 = None, t2 = None, n = None, norm = 0, method = "corr", count = None,
                chunk_size = None, thread_divisor = None,  
                auto_background = False,  viewer = None, viewer_interval = 1, mode = "full", mask = None, stats = True):
    """Iterative version of :func:`ccorr`.
    
    Parameters
    ----------
    data : iterable
        An iterable object, iterating over dual-frame ndarray data.
    t1 : int or array-like, optional
        Array of integers defining frame times of the first data. If it is a scalar
        it defines the length of the input data
    t2 : array-like, optional
        Array of integers defining frame times of the second data. If not provided, 
        regular time-spaced data is assumed.  
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3 (default).
    method : str, optional
        Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    count : int, optional
        If given, it defines how many elements of the data to process. If not given,
        count is set to len(t1) if that is not specified, it is set to len(data).     
    chunk_size : int
        Length of data chunk. 
    thread_divisor : int, optional
        If specified, input frame is reshaped to 2D with first axis of length
        specified with the argument. It defines how many treads are run. This
        must be a divisor of the total size of the frame. Using this may speed 
        up computation in some cases because of better memory alignment and
        cache sizing.
    auto_background : bool
        Whether to use data from first chunk to calculate and subtract background.
    viewer : any, optional
        You can use :class:`.viewer.MultitauViewer` to display data.
    viewer_interval : int, optional
        A positive integer, defines how frequently are plots updated 1 for most 
        frequent, higher numbers for less frequent updates. 
    mode : str
        Either "full" or "partial". With mode = "full", output of this function 
        is identical to the output of :func:`ccorr_multi`. With mode = "partial", 
        cross correlation between neigbouring chunks is not computed.
    mask : ndarray, optional
        If specifed, computation is done only over elements specified by the mask.
        The rest of elements are not computed, np.nan values are written to output
        arrays.
    stats : bool
        Whether to return stats as well.
        
    Returns
    -------
    ccorr_data, bg, var : ccorr_type, ndarray, ndarray
        Ccorr data, background and variance data. See :func:`ccorr` for definition 
        of accorr_type
    ccorr_data : ccorr_type
        If `stats` == False
    """
    
    if (t1 is None and t2 is not None) or (t2 is None and t1 is not None):
        raise ValueError("Both `t1` and `t2` arguments must be provided")
    
    period = 1
    nlevel = 0
    #in order to prevent circular import we import it here
    from cddm.multitau import _compute_multi_iter, _VIEWERS

    for i, mdata in enumerate(_compute_multi_iter(data, t1, t2, period = period , level_size = n , 
                        chunk_size = chunk_size,   method = method, count = count, auto_background = auto_background, nlevel = nlevel, 
                        norm = norm,  thread_divisor = thread_divisor, mode = mode, mask = mask,stats = stats, cross = True)):
        if stats == True:
            (data, dummy), bg, var = mdata
        else:
            data, dummy  = mdata
        
        if viewer is not None:
            if i == 0:
                _VIEWERS["ccorr"] = viewer
            if i%viewer_interval == 0:
                if stats == True:
                    viewer.set_data(data, background = bg, variance = var)
                    viewer.plot()
                else:
                    viewer.set_data(data)
                    viewer.plot()
    if stats == True:
        return data, bg, var
    else:
        return data

def iacorr(data, t = None, n = None, norm = 0, method = "corr", count = None,
                chunk_size = None, thread_divisor = None,  
                auto_background = False,  viewer = None, viewer_interval = 1, mode = "full", mask = None, stats = True):
    """Iterative version of :func:`ccorr`
        
    Parameters
    ----------
    data : iterable
        An iterable object, iterating over single-frame ndarray data.
    t : int or array-like, optional
        Array of integers defining frame times of the data. If it is a scalar
        it defines the length of the input data
    n : int, optional
        Determines the length of the output (max time delay - 1 by default). 
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3 (default).
    method : str, optional
        Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    chunk_size : int
        Length of data chunk. 
    count : int, optional
        If given, it defines how many elements of the data to process. If not given,
        count is set to len(t1) if that is not specified, it is set to len(data).  
    thread_divisor : int, optional
        If specified, input frame is reshaped to 2D with first axis of length
        specified with the argument. It defines how many treads are run. This
        must be a divisor of the total size of the frame. Using this may speed 
        up computation in some cases because of better memory alignment and
        cache sizing.
    auto_background : bool
        Whether to use data from first chunk to calculate and subtract background.
    viewer : any, optional
        You can use :class:`.viewer.MultitauViewer` to display data.
    viewer_interval : int, optional
        A positive integer, defines how frequently are plots updated 1 for most 
        frequent, higher numbers for less frequent updates. 
    mode : str
        Either "full" or "partial". With mode = "full", output of this function 
        is identical to the output of :func:`ccorr_multi`. With mode = "partial", 
        cross correlation between neigbouring chunks is not computed.
    mask : ndarray, optional
        If specifed, computation is done only over elements specified by the mask.
        The rest of elements are not computed, np.nan values are written to output
        arrays.
    stats : bool
        Whether to return stats as well.
        
    Returns
    -------
    acorr_data, bg, var : acorr_type, ndarray, ndarray
        Acorr data, background and variance data. See :func:`acorr` for definition 
        of acorr_type
    acorr_data : acorr_type
        If `stats` == False
    """

    period = 1
    nlevel = 0
    #in order to prevent circular import we import it here
    from cddm.multitau import _compute_multi_iter, _VIEWERS

    for i, mdata in enumerate(_compute_multi_iter(data, t, period = period , level_size = n , 
                        chunk_size = chunk_size,   method = method,  count = count, auto_background = auto_background, nlevel = nlevel, 
                        norm = norm,  thread_divisor = thread_divisor, mode = mode, mask = mask,stats = stats, cross = False)):
        if stats == True:
            (data, dummy), bg, var = mdata
        else:
            data, dummy  = mdata
        if viewer is not None:
            if i == 0:
                _VIEWERS["acorr"] = viewer
            if i%viewer_interval == 0:
                if stats == True:
                    viewer.set_data(data, background = bg, variance = var)
                    viewer.plot()
                else:
                    viewer.set_data(data)
                    viewer.plot()
    if stats == True:
        return data, bg, var
    else:
        return data    



