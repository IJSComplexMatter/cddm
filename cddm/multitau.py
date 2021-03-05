"""
Multiple-tau algorithm and log averaging functions.

Multitau analysis (in-memory, general data versions):

* :func:`ccorr_multi` to calculate cross-correlation/difference functions.
* :func:`acorr_multi` to calculate auto-correlation/difference functions.

For out-of-memory analysis and real-time calculation use:
 
* :func:`iccorr_multi` to calculate cross-correlation/difference functions
* :func:`iacorr_multi` to calculate auto-correlation/difference functions

For normalization and merging of the results of the above functions use:
    
* :func:`normalize_multi` to normalize the outputs of the above functions.
* :func:`log_merge` To merge results of the multi functions into one continuous data.

For time averaging of linear spaced data use:
 
* :func:`multilevel` to perform multi level averaging.
* :func:`merge_multilevel` to merge multilevel in a continuous log spaced data
* :func:`log_average` to perform both of these in one step.
"""

from __future__ import absolute_import, print_function, division

from cddm.core import normalize, ccorr,acorr, NORM_STRUCTURED, NORM_STANDARD, NORM_SUBTRACTED, NORM_WEIGHTED, NORM_COMPENSATED
from cddm.core import _transpose_data, _inspect_cross_arguments,_default_norm,\
        _is_aligned, _move_axis_and_align, _inspect_auto_arguments, abs2,\
        reshape_input, reshape_output, thread_frame_shape, reshape_frame
import numpy as np

from cddm.conf import CDTYPE, FDTYPE, IDTYPE
from cddm.print_tools import print_progress,  print_frame_rate, enable_prints, disable_prints
from cddm.print_tools import print1, print2

from cddm._core_nb import mean,  convolve, _calc_stats_vec

from cddm.norm import weighted_sum,weight_from_data, _method_from_data,_inspect_scale,_inspect_mode,_default_norm_from_data, scale_factor, norm_from_string
from cddm.avg import log_interpolate

import time

#: placehold for matplotlib plots so that they are not garbage collected diring live view
_VIEWERS = {}

BINNING_FIRST = 0
"""No binning (select every second element)"""
BINNING_MEAN = 1
"""Binning (take mean value)"""
BINNING_CHOOSE = 2
"""Binning with random selection"""

#for backward compatibility
BINNING_SLICE = BINNING_FIRST

def mean_data(data, axis = 0, out_axis = None,**kw):
    """Binning function. Takes data and performs channel binning over a specifed
    axis. If specified, also moves axis to out_axis."""
    if out_axis is None:
        out_axis = axis
    f = np.moveaxis(data,axis,-1)
    n = f.shape[-1] #this makes it possible to bin also odd-sized data.. skip one element
    f1 = np.moveaxis(f[...,0:n-1:2],-1,out_axis)
    f2 = np.moveaxis(f[...,1:n:2],-1,out_axis)
    out = np.empty(f1.shape, f1.dtype)
    return mean (f1, f2, out = out)

def choose_data(data, axis = 0, out_axis = None, r = None):
    """Instead of binning, this randomly selects data.
    If specified, also moves axis to out_axis."""
    if out_axis is None:
        out_axis = axis
    if r is None:
        r = np.random.rand()
    f = np.moveaxis(data,axis,-1)
    n = f.shape[-1] #this makes it possible to bin also odd-sized data.. skip one element
    f1 = np.moveaxis(f[...,0:n-1:2],-1,out_axis)
    f2 = np.moveaxis(f[...,1:n:2],-1,out_axis)
    out = np.empty(f1.shape, f1.dtype)
    if r < 0.5:
        out[...] = f2
    else:
        out[...] = f1
    return out
    #return  choose(f1,f2, out)

def slice_data(data, axis = 0, out_axis = None,**kw):
    """Slices data so that it takes every second channel. If specified, also moves axis to out_axis."""
    if out_axis is None:
        out_axis = axis
    f = np.moveaxis(data,axis,-1)
    return np.moveaxis(f[...,::2],-1,out_axis)

def _default_method_and_binning(method,binning):
    """inspects validity of mode and binning and sets binning default if not specified"""
    if method is None:
        method = "corr"
    
    if method in ("corr","fft"):
        if binning is None:
            binning =  BINNING_MEAN
    elif method  == "diff":
        if binning is None:
            binning =  BINNING_FIRST
    else:
        raise ValueError("Unknown method, must be 'corr', 'diff' or 'fft'")
    if method == "diff":
        supported = (BINNING_CHOOSE,BINNING_FIRST)
    else:
        supported = (BINNING_MEAN,BINNING_FIRST, BINNING_CHOOSE)

    if binning not in supported:
        raise ValueError("Binning mode {} not supported for method '{}'".format(binning,method))
    return method, binning
            
def _get_binning_function(binning):
    """returns binning function"""
    if binning == BINNING_MEAN:
        _bin = mean_data
    elif binning == BINNING_FIRST:
        _bin = slice_data   
    elif binning == BINNING_CHOOSE:
        _bin = choose_data
    return _bin

def _determine_lengths(length, n, period, nlevel):
    assert n > 4
    n_fast = period * n
    n_slow = n
    if nlevel is None:
        n_decades = 0
        while length// (2**(n_decades)) >= n:
            n_decades += 1
        nlevel = n_decades -1
    nlevel = int(nlevel)
    assert nlevel >= 0
    return n_fast, n_slow, nlevel

def _compute_multi(f1,f2 = None, t1 = None, t2 = None, axis = 0, period = 1, level_size = 2**4, 
                         binning = None,  nlevel = None, norm = None, method = None, align = False, thread_divisor = None, mask = None):
    """Implements multiple tau algorithm for cross(auto)-correlation(difference) calculation.
    """
    #initial time for computation time computation
    t0 = time.time()
    # is it cross or auto
    cross = False if f2 is None else True
    
    method, binning = _default_method_and_binning(method, binning)
    _bin = _get_binning_function(binning)
    
    norm = _default_norm(norm, method, cross)
    
    correlate = True if method in ("corr","fft") else False
        
    if cross:
        f1,f2,t1,t2,axis,level_size = _inspect_cross_arguments(f1,f2,t1,t2,axis,level_size, None)
    else:
        f1,t1,axis,level_size = _inspect_auto_arguments(f1,t1,axis,level_size, None)
    
    #reshape input data for faster threaded execution.
    f1, kshape = reshape_input(f1, axis = axis, thread_divisor = thread_divisor, mask = mask)
    if cross:
        f2, kshape = reshape_input(f2, axis = axis, thread_divisor = thread_divisor, mask = mask)

    new_axis = -2 if f1.ndim > 1 else -1    

    if t1 is None and t2 is None:
        if _is_aligned(f1, axis, align):
            new_axis = -1
     
    f1 = _move_axis_and_align(f1,axis,new_axis, align)
    f2 = _move_axis_and_align(f2,axis,new_axis, align) if cross else None

    axis = new_axis
    
    length = f1.shape[axis]
    
    n_fast, n_slow, nlevel = _determine_lengths(length,level_size,period,nlevel)
    
    if correlate == True:
        #we need to track square of the signal
        f1s = abs2(f1) if (norm & NORM_STRUCTURED) else None
        f2s = abs2(f2) if (norm & NORM_STRUCTURED) and cross else None
    else:
        f1s,f2s = None,None

    print1("Computing {}...".format("cross-correlation")) if cross else print1("Computing {}...".format("auto-correlation"))

    print2("   * period         : {}".format(period))
    print2("   * level_size     : {}".format(level_size))
    print2("   * binning        : {}".format(binning))
    print2("   * method         : {}".format(method))
    print2("   * nlevel         : {}".format(nlevel))
    print2("   * norm           : {}".format(norm))
    print2("   * align          : {}".format(align))
    print2("   * thread_divisor : {}".format(thread_divisor))
    print2("   * mask           : {}".format(mask is not None))

    
    print_progress(0, 100)        
    v = disable_prints()
    

    if cross == True:
        data_fast = ccorr(f1,f2,t1,t2, f1s = f1s, f2s = f2s, axis = axis, n = n_fast, norm = norm, method = method)
    else:
        data_fast = acorr(f1, t = t1, fs = f1s, axis = axis, n = n_fast, norm = norm, method = method)

    k_shape = data_fast[0].shape[0:-1]
    
    if axis == -1:
        new_axis = -1
        slow_shape = (nlevel,) + k_shape + (n_slow,)
    else:
        new_axis = axis
        slow_shape = (nlevel,) + k_shape[0:-1] + (n_slow,) + k_shape[-1:] 

    out_slow = np.zeros(slow_shape, FDTYPE)  
    out_slow = _transpose_data(out_slow, new_axis) 
        
    count_slow = np.zeros((nlevel, n_slow,),IDTYPE)
    m1, m2, sq = None, None, None
 
    if (norm & NORM_STRUCTURED) and correlate == True:
        sq = np.zeros(slow_shape, FDTYPE) 
        sq = _transpose_data(sq, new_axis) 
        
    if norm & NORM_SUBTRACTED:
        m1 = np.zeros(slow_shape, CDTYPE) 
        m1 = _transpose_data(m1, new_axis) 
        m2 = np.zeros(slow_shape, CDTYPE) if cross else None
        m2 = _transpose_data(m2, new_axis) if cross else None

    if correlate == True:
        out = data_fast, (out_slow, count_slow, sq, m1,m2)   
    else:
        out = data_fast, (out_slow, count_slow, m1,m2)  
    
    enable_prints(v) 
    
    progress = 0.5
    progress_int = int(100*progress)
    
    print_progress(progress_int, 100)        
          
    data_slow = out[1]
    
    #do multi-tau caluclation

    for i in range(nlevel):
        f1 = _bin(f1,axis, new_axis)
        f2 = _bin(f2,axis, new_axis) if cross else None
        if (norm & NORM_STRUCTURED) and correlate:
            f1s = _bin(f1s,axis, new_axis)
            f2s = _bin(f2s,axis, new_axis) if cross else None
        axis = new_axis
                
        _out =  tuple(((d[i] if d is not None else None) for d in data_slow))
        
        v = disable_prints() 

        if cross == True:
            ccorr(f1,f2, f1s = f1s, f2s = f2s,axis = axis, n = n_slow, norm = norm, aout = _out, method = method)
        else:
            acorr(f1, fs = f1s,axis = axis, norm = norm, n = n_slow, aout = _out, method = method)

        enable_prints(v)
        progress = progress*0.5
        progress_int = progress_int + int(100*progress)
        
        print_progress(progress_int, 100)          
         
    print_progress(100, 100) 
    
    print_frame_rate(length, t0)
    
    #reshape back to original data shape
    return tuple((reshape_output(o, kshape, mask = mask) for o in out))


def ccorr_multi(f1,f2, t1 = None, t2 = None,  level_size = 2**4, norm = None, method = None, align = False, axis = 0,
                period = 1, binning = None,  nlevel = None,  thread_divisor = None, mask = None):
    """Multitau version of :func:`.core.ccorr`
        
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
    level_size : int, optional
        If provided, determines the length of the output.
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
    period : int, optional
        Period of the irregular-spaced random triggering sequence. For regular
        spaced data, this should be set to 1 (deefault).
    binning : int, optional
        Binning mode (0 - no binning, 1 : average, 2 : random select)
    nlevel : int, optional
        If specified, defines how many levels are used in multitau algorithm.
        If not provided, all available levels are used.
    thread_divisor : int, optional
        If specified, input frame is reshaped to 2D with first axis of length
        specified with the argument. It defines how many treads are run. This
        must be a divisor of the total size of the frame. Using this may speed 
        up computation in some cases because of better memory alignment and
        cache sizing.
        
    Returns
    -------
    lin, multi : ccorr_type, ccorr_type
        A tuple of linear (short delay) data and multilevel (large delay) data
        See :func:`.core.ccorr` for definition of ccorr_type
    """
    return _compute_multi(f1,f2, t1,t2, axis = axis, period = period, level_size = level_size, align = align,
                         binning = binning,  nlevel = nlevel, method = method, norm = norm,thread_divisor = thread_divisor, mask = mask)

def acorr_multi(f, t = None,  level_size = 2**4, norm = None, method = None, align = False, axis = 0,
                period = 1, binning = None,  nlevel = None,  thread_divisor = None, mask = None):
    """Multitau version of :func:`.core.acorr`
        
    Parameters
    ----------
    f : array-like
        A complex ND array
    t : array-like, optional
        Array of integers defining frame times of the data. If not provided, 
        regular time-spaced data is assumed.
    level_size : int, optional
        If provided, determines the length of the output.
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
    period : int, optional
        Period of the irregular-spaced random triggering sequence. For regular
        spaced data, this should be set to 1 (deefault).
    binning : int, optional
        Binning mode (0 - no binning, 1 : average, 2 : random select)
    nlevel : int, optional
        If specified, defines how many levels are used in multitau algorithm.
        If not provided, all available levels are used.
    thread_divisor : int, optional
        If specified, input frame is reshaped to 2D with first axis of length
        specified with the argument. It defines how many treads are run. This
        must be a divisor of the total size of the frame. Using this may speed 
        up computation in some cases because of better memory alignment and
        cache sizing.
        
    Returns
    -------
    lin, multi : acorr_type, acorr_type
        A tuple of linear (short delay) data and multilevel (large delay) data
        See :func:`.core.acorr` for definition of acorr_type
    """
    
    return _compute_multi(f, t1 = t, axis = axis, period = period, level_size = level_size, align = align,
                         binning = binning,  nlevel = nlevel, method = method, norm = norm,thread_divisor = thread_divisor, mask = mask)
    
def _add_data1(i,x, out, binning = True, r = None):
    chunk_size = out.shape[-2]
    n_decades = out.shape[0]
    
    index_top = i % chunk_size
        
    out[0,...,index_top,:] = x 
    
    for j in range(1, n_decades):
        if (i+1) % (2**j) == 0:
            index_low = (i//(2**j)) % (chunk_size)
            
            if binning == BINNING_MEAN:
                x = mean(x, out[j-1,...,index_top-1,:], out[j,...,index_low,:])
            elif binning == BINNING_CHOOSE:
                if r is None:
                    r = np.randomd.rand()
                x = x if r > 0.5 else out[j-1,...,index_top-1,:]
                out[j,...,index_low,:] = x
                #x = choose(x, out[j-1,...,index_top-1,:], out[j,...,index_low,:])
            else:
                out[j,...,index_low,:] = x
            
            index_top = index_low
        else:
            break

def _add_data2(i,x1, x2, out1, out2, binning = True, r = None):
    chunk_size = out1.shape[-2]
    n_decades = out1.shape[0]
    
    index_top = i % chunk_size
        
    out1[0,...,index_top,:] = x1
    out2[0,...,index_top,:] = x2   
    
    for j in range(1, n_decades):
        if (i+1) % (2**j) == 0:
            index_low = (i//(2**j)) % (chunk_size)
            
            if binning == BINNING_MEAN:
                x1 = mean(x1, out1[j-1,...,index_top-1,:], out1[j,...,index_low,:])
                x2 = mean(x2, out2[j-1,...,index_top-1,:], out2[j,...,index_low,:])  
            elif binning == BINNING_CHOOSE:
                if r is None:
                    r = np.randomd.rand()
                x1 = x1 if r > 0.5 else out1[j-1,...,index_top-1,:]
                x2 = x2 if r > 0.5 else out2[j-1,...,index_top-1,:]
                out1[j,...,index_low,:] = x1
                out2[j,...,index_low,:] = x2
                #x1 = choose(x1, out1[j-1,...,index_top-1,:], out1[j,...,index_low,:])
                #x2 = choose(x2, out2[j-1,...,index_top-1,:], out2[j,...,index_low,:])  
            else:
                out1[j,...,index_low,:] = x1
                out2[j,...,index_low,:] = x2
            
            index_top = index_low
        else:
            break

def _get_frames(d, cross = False, mask = None):
    """gets two frames (or one frame and NoneType) from dual (or single) frame data"""
    if cross == True:
        x1,x2 = d
        if mask is not None:
            x1,x2 = x1[mask],x2[mask]
    else:
        if len(d) == 1 and isinstance(d, tuple):
            x1, = d
        else:
            x1 = d
        if mask is not None:
            x1 = x1[mask]
        x2 = None
    return x1, x2

def _default_chunk_size(chunk_size, count, level_size):
    if level_size > count:
        raise ValueError("'level_size' parameter is larger than 'count'")
    if chunk_size is None:
        #start with count
        chunk_size = count
        while (chunk_size//2) >= level_size and chunk_size > 128: 
            chunk_size = chunk_size//2

    if chunk_size > count:
        raise ValueError("'chunk_size' is larger than 'count'")
    if chunk_size < level_size:
        raise ValueError("'chunk_size' is smaller than 'level_size'")
    return chunk_size

def _inspect_t1_t2_count(t1,t2,count,data):
    if t1 is None:
        if count is None:
            try:
                count = len(data)
            except TypeError:
                raise ValueError("You must define `count`")
        t1 = np.arange(int(count))
    else:
        t1 = np.asarray(t1)  
    if count is None:
        count = len(t1)
    count = int(count)
    t2 = np.asarray(t2) if t2 is not None else t1 
    if count > len(t1):
        raise ValueError("`count` is too large for given time arrays")
    if len(t1) != len(t2):
        raise ValueError("Lengths of time arrays do not match")
    return t1,t2,count

def _compute_multi_iter(data, t1, t2 = None, period = 1, level_size = 16, 
                        chunk_size = None,  binning = None, method = "corr", count = None,
                        auto_background = False, nlevel = None,  norm = None,
                        stats = False, thread_divisor = None, mode = "full", mask = None, cross = True):
    t0 = time.time()
    
    if mode not in ("full", "chunk"):
        raise ValueError("Unknown mode '{}'".format(mode))
    
    method, binning = _default_method_and_binning(method,binning)
    t1,t2, count = _inspect_t1_t2_count(t1,t2,count,data)
    correlate = False if method == "diff" else True
        
    level_size = int(level_size)
   
    chunk_size = _default_chunk_size(chunk_size,count,level_size)
    
    half_chunk_size = chunk_size
    chunk_size = 2* half_chunk_size #this will be the actual chunk size of data, split into two halfs
       
    if level_size > half_chunk_size: 
        raise ValueError("`chunk_size` must be even and greater or equal to `level_size`")
    
    n_fast = int(period) * level_size
    n_slow = level_size
    
    n_decades = 0
    while count//(chunk_size * 2**(n_decades)) > 0:
        n_decades += 1  
    
    nframes = int(2**(n_decades-1) * chunk_size) #int in case n_decades = 0
    
    if nframes != count:
        print("WARNING: chunk_size not compatible with the length of the input data, data length cropped to {}".format(nframes))
    
    n_decades_max = 0
    while nframes// (2**(n_decades_max)) >= level_size:
        n_decades_max += 1 
             
    if nlevel is None:
        nlevel = n_decades_max -1
    else:
        n_decades = min(nlevel + 1, n_decades_max)
        if n_decades < 1:
            raise ValueError("Invalid nlevel")
        nlevel = min(int(nlevel),n_decades_max -1)
              
    t_slow = np.arange(nframes)
          
    i = 0 #initialize to zero, in case iterator is empty
    
    for i,d in enumerate(data):
        if i == nframes:
            break

        #first and second frame or masked frames, x2 is None if cross = False
        x1, x2 = _get_frames(d, cross, mask)

        if i == 0:
            #do initialization
            original_shape = x1.shape
            force_2d = True if mask is None else False
            shape = thread_frame_shape(original_shape, thread_divisor, force_2d)

            norm = _default_norm(norm, method, cross)
            
            print1("Computing {}...".format("cross-correlation")) if cross else print1("Computing {}...".format("auto-correlation"))
            
            print2("   * length          : {}".format(nframes))
            print2("   * count           : {}".format(count))
            print2("   * period          : {}".format(period))
            print2("   * chunk_size      : {}".format(half_chunk_size))
            print2("   * level_size      : {}".format(level_size))
            print2("   * binning         : {}".format(binning))
            print2("   * method          : {}".format(method))
            print2("   * auto_background : {}".format(auto_background))
            print2("   * nlevel          : {}".format(nlevel))
            print2("   * norm            : {}".format(norm))
            print2("   * mode            : {}".format(mode))
            print2("   * mask            : {}".format(mask is not None))
            print2("   * thread_divisor  : {}".format(thread_divisor))
            print2("   * stats           : {}".format(stats))
            
            print_progress(0, nframes)

            
            fast_shape = shape[0:-1] + (n_fast,) + shape[-1:]
            
            out_fast = np.zeros(fast_shape, FDTYPE)
            out_fast = _transpose_data(out_fast)
            
            if norm & NORM_SUBTRACTED:
                m1_fast = np.zeros(fast_shape, CDTYPE)
                m1_fast = _transpose_data(m1_fast)                  
                m2_fast = np.zeros(fast_shape, CDTYPE) if cross  == True else None
                m2_fast = _transpose_data(m2_fast) if cross  == True else None
            else:
                m1_fast, m2_fast = None, None
                
            if (norm & NORM_STRUCTURED) and correlate == True:
                sq_fast = np.zeros(fast_shape, FDTYPE)
                sq_fast = _transpose_data(sq_fast) 
            else:
                sq_fast = None
                
            count_fast = np.zeros((n_fast,),IDTYPE)
            
            if correlate:
                data_fast = out_fast, count_fast, sq_fast, m1_fast, m2_fast
            else:
                data_fast = out_fast, count_fast, m1_fast, m2_fast
            
            slow_shape = (nlevel,) + shape[0:-1] + (n_slow,) + shape[-1:]
            
            
            out_slow = np.zeros(slow_shape, FDTYPE)  
            out_slow = _transpose_data(out_slow)    
            
            if norm & NORM_SUBTRACTED:
                m1_slow = np.zeros(slow_shape, CDTYPE)  
                m1_slow = _transpose_data(m1_slow)                  
                m2_slow = np.zeros(slow_shape, CDTYPE) if cross  == True else None
                m2_slow = _transpose_data(m2_slow) if cross  == True else None
            else:
                m1_slow, m2_slow = None, None
            if (norm & NORM_STRUCTURED) and correlate == True:
                sq_slow = np.zeros(slow_shape, FDTYPE)  
                sq_slow = _transpose_data(sq_slow) 
            else:
                sq_slow = None
                
            count_slow = np.zeros((nlevel, n_slow,),IDTYPE)
            
            if n_decades > 0:
                data_shape = (n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:]
            else:
                data_shape = (1,) + shape[0:-1] + (half_chunk_size,) + shape[-1:]

            fdata1 = np.empty(data_shape, CDTYPE)
            fdata2 = np.empty(data_shape, CDTYPE) if cross  == True else None
            
            if correlate:
                data_slow = out_slow, count_slow, sq_slow, m1_slow, m2_slow
            else:
                data_slow = out_slow, count_slow, m1_slow, m2_slow
                     
            
            if (norm & NORM_STRUCTURED) and correlate == True:
                #allocate memory for square data
                sq1 = np.empty(data_shape, FDTYPE)
                sq2 = np.empty(data_shape, FDTYPE) if cross  == True else None
            
            if stats == True:
                sum1 = np.zeros(shape, CDTYPE)
                sum2 =  np.zeros(shape, CDTYPE) if cross  == True else None
                sqsum1 = np.zeros(shape, FDTYPE)
                sqsum2 = np.zeros(shape, FDTYPE) if cross  == True else None
                out_bg1 = np.empty(shape, CDTYPE)
                out_bg2 = np.empty(shape, CDTYPE) if cross  == True else None
                out_var1 = np.empty(shape, FDTYPE)
                out_var2 = np.empty(shape, FDTYPE) if cross  == True else None  
            bg1, bg2 = None, None
            
        
        x1 = x1.reshape(shape) 
        if cross == True:
            x2 = x2.reshape(shape)
            
        if i < half_chunk_size and auto_background == True:
            fdata1[0,...,i,:] = x1
            if cross:
                fdata2[0,...,i,:] = x2   

            if i == half_chunk_size -1:

                bg1 = fdata1[0,...,0:half_chunk_size,:].mean(-2)
                bg2 = fdata2[0,...,0:half_chunk_size,:].mean(-2) if cross  == True  else None
             
                
                for i in range(half_chunk_size):
                    r = np.random.rand()
                    x1 = fdata1[0,...,i,:] - bg1
                    x2 = fdata2[0,...,i,:] - bg2 if cross == True else None
                    if cross == True:
                        _add_data2(i,x1, x2, fdata1,fdata2, binning, r)
                        if (norm & NORM_STRUCTURED) and correlate:
                            _add_data2(i,abs2(x1), abs2(x2), sq1,sq2, binning, r) 
                    else:
                        _add_data1(i,x1, fdata1, binning,r)
                        if (norm & NORM_STRUCTURED) and correlate:
                            _add_data1(i,abs2(x1),sq1,binning,r)
        else:
            if bg1 is not None:
                x1 = x1 - bg1
            if bg2 is not None and cross:
                x2 = x2 - bg2
                
            r = np.random.rand() 
            if cross == True:
                _add_data2(i,x1, x2, fdata1,fdata2, binning,r)
                if (norm & NORM_STRUCTURED) and correlate:
                    _add_data2(i,abs2(x1), abs2(x2), sq1,sq2, binning,r)
            else:
                _add_data1(i,x1, fdata1,binning,r)
                if (norm & NORM_STRUCTURED) and correlate:
                    _add_data1(i,abs2(x1),sq1,binning,r)                
                
        if i % (half_chunk_size) == half_chunk_size -1: 

            ichunk = i//half_chunk_size
            
            fstart1 = half_chunk_size * (ichunk%2)
            fstop1 = fstart1 + half_chunk_size

            fstart2 = half_chunk_size * ((ichunk-1)%2)
            fstop2 = fstart2 + half_chunk_size
            
            istart1 = ichunk * half_chunk_size
            istop1 = istart1 + half_chunk_size
            
            istart2 = istart1 - half_chunk_size
            istop2 = istop1 - half_chunk_size            


            if stats == True:
                _calc_stats_vec(fdata1[0,...,fstart1:fstop1,:], sum1, sqsum1)
                _calc_stats_vec(fdata2[0,...,fstart1:fstop1,:], sum2, sqsum2) if cross else None
                
                #divisor for normalization (mean value calculation)
                _divisor =  (ichunk + 1) * half_chunk_size
                
                np.divide(sum1,_divisor  , out_bg1)
                np.divide(sum2, _divisor , out_bg2) if cross else None
                np.divide(sqsum1, _divisor , out_var1)
                np.divide(sqsum2, _divisor , out_var2)  if cross else None
                np.subtract(out_var1, abs2(out_bg1), out_var1)
                np.subtract(out_var2, abs2(out_bg2), out_var2)  if cross else None
                               
            f1s = sq1[0,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None
            f2s = sq2[0,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate and cross else None
            
            out = data_fast
                
            verbosity = disable_prints() #we do not want to print messages of f()

            if cross:
                ccorr(fdata1[0,...,fstart1:fstop1,:],fdata2[0,...,fstart1:fstop1,:],t1[istart1:istop1],t2[istart1:istop1],f1s = f1s, f2s = f2s,axis = -2, n = n_fast, norm = norm,aout = out, method = method) 
            else:
                acorr(fdata1[0,...,fstart1:fstop1,:],t1[istart1:istop1],fs = f1s, axis = -2, n = n_fast, norm = norm, aout = out, method = method) 

            if istart2 >= 0 and mode == "full":
                
                f1s = sq1[0][...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None
                
                if cross:
                    f2s = sq2[0][...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                    ccorr(fdata1[0][...,fstart1:fstop1,:],fdata2[0][...,fstart2:fstop2,:],t1[istart1:istop1],t2[istart2:istop2],f1s = f1s, f2s = f2s, axis = -2, n = n_fast,norm = norm,aout = out, method = method) 
                else:
                    f2s = sq1[0][...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                    if correlate:
                        out = data_fast[0],data_fast[1], data_fast[2], None, None 
                    else:
                        out = data_fast[0],data_fast[1], None, None 
                    out = ccorr(fdata1[0][...,fstart1:fstop1,:],fdata1[0][...,fstart2:fstop2,:],t1[istart1:istop1],t1[istart2:istop2],f1s = f1s, f2s = f2s,  axis = -2, n = n_fast,norm = norm,aout = out, method = method)
                    if m1_fast is not None:
                        m1_fast += out[3] /2
                        m1_fast += out[4] /2
                if cross:
                    f1s = sq1[0][...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                    f2s = sq2[0][...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None  
                    ccorr(fdata1[0][...,fstart2:fstop2,:],fdata2[0][...,fstart1:fstop1,:],t1[istart2:istop2],t2[istart1:istop1],f1s = f1s, f2s = f2s,axis = -2, n = n_fast,norm = norm,aout = out, method = method) 
  

            for j in range(1, n_decades):

                
                if i % (half_chunk_size * 2**j) == half_chunk_size * 2**j -1: 
                    
                    ichunk = i//(half_chunk_size * 2**j)
                    
                    fstart1 = half_chunk_size * (ichunk%2)
                    fstop1 = fstart1 + half_chunk_size
        
                    fstart2 = half_chunk_size * ((ichunk-1)%2)
                    fstop2 = fstart2 + half_chunk_size
                    
                    istart1 = ichunk * half_chunk_size
                    istop1 = istart1 + half_chunk_size
                    
                    istart2 = istart1 - half_chunk_size
                    istop2 = istop1 - half_chunk_size  
                
  
                    f1s = sq1[j,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None
                    f2s = sq2[j,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate and cross else None  
                    
                    out =  tuple(((d[j-1] if d is not None else None) for d in data_slow))
                    
                    if cross:
                        ccorr(fdata1[j,...,fstart1:fstop1,:],fdata2[j,...,fstart1:fstop1,:],f1s = f1s, f2s = f2s, axis = -2, norm = norm, n = n_slow, aout = out, method = method)
                    else:
                        acorr(fdata1[j,...,fstart1:fstop1,:],fs = f1s, axis = -2, norm = norm, n = n_slow, aout = out, method = method)
        
                    if istart2 >= 0 and mode == "full":
                        f1s = sq1[j,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None
                        
                        if cross:
                            f2s = sq2[j,...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                            ccorr(fdata1[j,...,fstart1:fstop1,:],fdata2[j,...,fstart2:fstop2,:],t_slow[istart1:istop1],t_slow[istart2:istop2],f1s = f1s, f2s = f2s, axis = -2, n = n_fast,norm = norm,aout = out, method = method) 
                        else:
                            f2s = sq1[j,...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                            _out =  tuple(((d[j-1] if d is not None else None) for d in data_slow))
                            if correlate:
                                out = _out[0],_out[1], _out[2], None, None 
                            else:
                                out = _out[0],_out[1], None, None 
                            out = ccorr(fdata1[j,...,fstart1:fstop1,:],fdata1[j,...,fstart2:fstop2,:],t_slow[istart1:istop1],t_slow[istart2:istop2],f1s = f1s, f2s = f2s,  axis = -2, n = n_fast,norm = norm,aout = out, method = method)
                            if m1_slow is not None:
                                m1_slow[j-1] += out[3] /2
                                m1_slow[j-1] += out[4] /2
                        if cross:
                            f1s = sq1[j,...,fstart2:fstop2,:] if (norm & NORM_STRUCTURED) and correlate else None
                            f2s = sq2[j,...,fstart1:fstop1,:] if (norm & NORM_STRUCTURED) and correlate else None  
                            ccorr(fdata1[j,...,fstart2:fstop2,:],fdata2[j,...,fstart1:fstop1,:],t_slow[istart2:istop2],t_slow[istart1:istop1],f1s = f1s, f2s = f2s,axis = -2, n = n_fast,norm = norm,aout = out, method = method) 

                else:
                    break
                
            #are there still some data to process?

            out = data_fast, data_slow
                
            enable_prints(verbosity)#enable prints back to show progress
            print_progress(i+1, nframes)
            
            if stats == True:
                if cross:
                    yield tuple((reshape_output(o, original_shape,mask) for o in out)), (reshape_frame(out_bg1,original_shape,mask), reshape_frame(out_bg2,original_shape,mask)), (reshape_frame(out_var1,original_shape,mask), reshape_frame(out_var2,original_shape,mask))
                else:
                    yield tuple((reshape_output(o, original_shape,mask) for o in out)), reshape_frame(out_bg1,original_shape,mask),  reshape_frame(out_var1,original_shape,mask)

            else:
                yield tuple((reshape_output(o, original_shape,mask) for o in out))
    
    if i < nframes-1:
        print (i)
        raise ValueError("Data iterator too short.") 
         
    verbosity = disable_prints()
    
    #process rest of the data
    
    f1 = fdata1[-1]
    f2 = fdata2[-1] if cross else None
    if (norm & NORM_STRUCTURED) and correlate:
        sq1 = sq1[-1]
        sq2 = sq2[-1] if cross else None
        
    _bin = _get_binning_function(binning)
         
    for j in range(max(n_decades,1),nlevel+1):
        r = np.random.rand()
        f1 = _bin(f1,-2, r = r)
        f2 = _bin(f2,-2, r = r) if cross else None
        sq1 = _bin(sq1,-2, r = r) if (norm & NORM_STRUCTURED) and correlate else None
        sq2 = _bin(sq2,-2, r = r) if (norm & NORM_STRUCTURED) and correlate and cross else None
      
        out =  tuple(((d[j-1] if d is not None else None) for d in data_slow))
        
        if cross:
            ccorr(f1,f2, f1s = sq1, f2s = sq2, axis = -2, norm = norm,n = n_slow, aout = out, method = method)
        else:
            acorr(f1, fs = sq1, axis = -2, norm = norm,n = n_slow, aout = out, method = method)

    enable_prints(verbosity)

    out =  data_fast, data_slow
    print_frame_rate(nframes, t0)
    if stats == True:
        if cross:
            yield tuple((reshape_output(o, original_shape,mask) for o in out)), (reshape_frame(out_bg1,original_shape,mask), reshape_frame(out_bg2,original_shape,mask)), (reshape_frame(out_var1,original_shape,mask), reshape_frame(out_var2,original_shape,mask))
        else:
            yield tuple((reshape_output(o, original_shape,mask) for o in out)), reshape_frame(out_bg1,original_shape,mask),  reshape_frame(out_var1,original_shape,mask)
    
    else:
        yield tuple((reshape_output(o, original_shape,mask) for o in out))

def iccorr_multi(data, t1 = None, t2 = None, level_size = 2**4, norm = None, method = "corr", count = None, period = 1,
                 binning = None, nlevel = None, chunk_size = None, thread_divisor = None,  
                 auto_background = False,  viewer = None, viewer_interval = 1, mode = "full", mask = None, stats = True):
    """Iterative version of :func:`.ccorr_multi`
        
    Parameters
    ----------
    data : iterable
        An iterable object, iterating over dual-frame ndarray data.
    t1 : array-like, optional
        Array of integers defining frame times of the first data.  If not defined, 
        regular-spaced data is assumed.
    t2 : array-like, optional
        Array of integers defining frame times of the second data. If t1 is defined,
        you must define t2 as well.
    level_size : int, optional
        If provided, determines the length of the multi_level data.
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3 (default).
    method : str, optional
        Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    count : int, optional
        If given, it defines how many elements of the data to process. If not given,
        count is set to len(t1) if that is not specified, it is set to len(data).  
    period : int, optional
        Period of the irregular-spaced random triggering sequence. For regular
        spaced data, this should be set to 1 (deefault).
    binning : int, optional
        Binning mode (0 - no binning, 1 : average, 2 : random select)
    nlevel : int, optional
        If specified, defines how many levels are used in multitau algorithm.
        If not provided, all available levels are used.
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
        cross correlation between neighbouring chunks is not computed.
    mask : ndarray, optional
        If specifed, computation is done only over elements specified by the mask.
        The rest of elements are not computed, np.nan values are written to output
        arrays.
    stats : bool
        Whether to return stats as well.
        
    Returns
    -------
    (lin, multi), bg, var : (ccorr_type, ccorr_type), ndarray, ndarray
        A tuple of linear (short delay) data and multilevel (large delay) data,
        background and variance data. See :func:`.core.ccorr` for definition 
        of ccorr_type
    lin, multi : ccorr_type, ccorr_type
        If `stats` == False
    """
    if (t1 is None and t2 is not None) or (t2 is None and t1 is not None):
        raise ValueError("Both `t1` and `t2` arguments must be provided")
    
    for i, data in enumerate(_compute_multi_iter(data, t1, t2, period = period, level_size = level_size , 
                        chunk_size = chunk_size,  binning = binning,  method = method, count = count, auto_background = auto_background,
                        nlevel = nlevel, norm = norm, stats = stats,mask = mask, thread_divisor = thread_divisor, cross = True)):
        if viewer is not None:
            if i == 0:
                _VIEWERS["ccorr_multi"] = viewer
            if i%viewer_interval == 0:
                if stats == True:
                    viewer.set_data(data[0], background = data[1], variance = data[2])
                    viewer.plot()
                else:
                    viewer.set_data(data)
                    viewer.plot()             
    return data


def iacorr_multi(data, t = None, level_size = 2**4, norm = None, method = "corr", count = None, period = 1,
                 binning = None, nlevel = None, chunk_size = None, thread_divisor = None,  
                 auto_background = False,  viewer = None, viewer_interval = 1, mode = "full", mask = None, stats = True):
    """Iterative version of :func:`.acorr_multi`
        
    Parameters
    ----------
    data : iterable
        An iterable object, iterating over dual-frame ndarray data.
    t : array-like, optional
        Array of integers defining frame times of the first data. If not defined, 
        regular-spaced data is assumed.
    level_size : int, optional
        If provided, determines the length of the multi_level data.
    norm : int, optional
        Specifies normalization procedure 0,1,2, or 3 (default).
    method : str, optional
        Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on 
        the rest of the input parameters.
    count : int, optional
        If given, it defines how many elements of the data to process. If not given,
        count is set to len(t1) if that is not specified, it is set to len(data).  
    period : int, optional
        Period of the irregular-spaced random triggering sequence. For regular
        spaced data, this should be set to 1 (deefault).
    binning : int, optional
        Binning mode (0 - no binning, 1 : average, 2 : random select)
    nlevel : int, optional
        If specified, defines how many levels are used in multitau algorithm.
        If not provided, all available levels are used.
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
        Either "full" or "chunk". With mode = "full", output of this function 
        is identical to the output of :func:`ccorr_multi`. With mode = "chunk", 
        cross correlation between neighbouring chunks is not computed.
    mask : ndarray, optional
        If specifed, computation is done only over elements specified by the mask.
        The rest of elements are not computed, np.nan values are written to output
        arrays.
    stats : bool
        Whether to return stats as well.
        
    Returns
    -------
    (lin, multi), bg, var : (acorr_type, acorr_type), ndarray, ndarray
        A tuple of linear (short delay) data and multilevel (large delay) data,
        background and variance data. See :func:`.core.acorr` for definition 
        of acorr_type
    lin, multi : acorr_type, acorr_type
        If `stats` == False
    """
    for i, data in enumerate(_compute_multi_iter(data, t, None, period = period, level_size = level_size , 
                        chunk_size = chunk_size,  binning = binning,  method = method, count = count,auto_background = auto_background,
                        nlevel = nlevel, norm = norm, stats = stats, thread_divisor = thread_divisor, mask = mask, cross = False)):
        if viewer is not None:
            if i == 0:
                _VIEWERS["acorr_multi"] = viewer
            if i%viewer_interval == 0:
                if stats == True:
                    viewer.set_data(data[0], background = data[1], variance = data[2])
                    viewer.plot()
                else:
                    viewer.set_data(data)
                    viewer.plot()
    return data

def triangular_kernel(n):
    """Returns normalized triangular kernel for a given level integer
    
    Examples
    --------
    >>> triangular_kernel(0) #zero-th level
    array([1.])
    >>> triangular_kernel(1) #first level
    array([0.25, 0.5 , 0.25])
    """
    size = 1
    power = 2**n
    for i in range(n):
        size = size*2 +1
    out = np.zeros(shape = (size,))
    
    for i in range(size//2):
        out[i] = (i+1)/power/power
        out[-i-1] = out[i]
    out[size//2] = 1/power
    return out

def multilevel_count(count, level_size, binning = BINNING_MEAN):
    """Returns effective number of measurements in multilevel data calculated
    from linear time-space count data using :func:`multilevel`
    
    This function can be used to estimate the effective number of realizations
    of each data point in multilevel data for error estimation.
    
    Parameters
    ----------
    count : ndarray
        Count data,
    level_size : int
        Level size used in multilevel averaging.
    binning : int
        Binning mode that was used in :func:`multilevel`
        
    Returns
    -------
    x  : ndarray
        Multilevel effective count array. Shape of this data depends on the 
        length of the original data and the provided level_size parameter.
    """
    binning = int(binning)
    #kernel = triangular_kernel(n)
    #return 1/(kernel**2).sum()

    size = int(level_size)
    if size < 2:
        raise ValueError("level_size must be greater than 1")
    #determine multitau level (number of decades)
    level = 0
    while size * 2**(level) < count.shape[-1]:
        level +=1
    
    shape = (level+1,size)
    
    out = np.empty(shape = shape, dtype = FDTYPE)
    
    count2 = count **2
    
    for l in range(level+1):
        if binning == BINNING_MEAN:
            kernel = triangular_kernel(l)
        else:
            kernel = np.array([1.]) #np.array([1./2**l])
        kernel2 = kernel**2
        
        data = np.convolve(kernel2, count2, "same")
        norm = np.convolve(kernel, count , "same")
        data_sliced = data[0::2**l][0:size]
        norm_sliced = norm[0::2**l][0:size]
        n_sliced = data_sliced.shape[-1]
        out[l,0:n_sliced] = norm_sliced**3 / data_sliced
        
        #missing data... fill nans
        if n_sliced != size:
            out[l,n_sliced:] = np.nan
        
    return out

def ccorr_multi_count(count, period = 1, level_size = 16, binning = True):
    """Returns an effective count data for the equivalent multilevel cross-correlation
    calculation. 
    
    Parameters
    ----------
    count : int
        Number of frames processed
    period : int
        The period argument used in correlation calculation
    level_size : int
        Level size used in correlation calculation
    binning, int
        Binning mode used in correlation calculation
        
    Returns
    -------
    cf,cs : ndarray, ndarray
        Count arrays for the linear part and the multilevel part of the multilevel
        correlation calculation.
    """
    count_fast = np.linspace(count/period*2, (count - level_size)/period*2 , period*level_size)
    count_slow = np.linspace(count,1,count)*2
    count_slow = multilevel_count(count_slow, level_size, binning)  
    if binning != BINNING_MEAN:
        n = count_slow.shape[0]
        c = 2**(np.arange(n))
        count_slow = count_slow/c[:,None]
    
    return count_fast, count_slow[1:]

def acorr_multi_count(count, period = 1, level_size = 16, binning = True):
    """Returns an effective count data for the equivalent multilevel auto-correlation
    calculation. 
    
    Parameters
    ----------
    count : int
        Number of frames processed
    period : int
        The period argument used in correlation calculation
    level_size : int
        Level size used in correlation calculation
    binning, int
        Binning mode used in correlation calculation
        
    Returns
    -------
    cf,cs : ndarray, ndarray
        Count arrays for the linear part and the multilevel part of the multilevel
        correlation calculation,  
    """ 
    
    count_fast = np.linspace(count/period, (count - level_size)/period , period*level_size)
    count_fast[0:period+1] = np.linspace(0,count_fast[period], period +1)
    count_slow = np.linspace(count,1,count)
    count_slow = multilevel_count(count_slow, level_size, binning)  
    if binning != BINNING_MEAN:
        n = count_slow.shape[0]
        c = 2**(np.arange(n))
        count_slow = count_slow/c[:,None]
        
    return count_fast, count_slow[1:]

def _inspect_binning(binning):
    if binning not in (BINNING_CHOOSE, BINNING_MEAN, BINNING_FIRST):
        raise ValueError("Unsupported binning mode")
    return binning
    
def multilevel(data, level_size = 16, binning = BINNING_MEAN):
    """Computes a multi-level version of the linear time-spaced data.
    
    Parameters
    ----------
    data : ndarray
        Normalized correlation data 
    level_size : int
        Level size
    binning : int
        Binning mode, either BINNING_FIRST or BINNING_MEAN (default).
        
    Returns
    -------
    x  : ndarray
        Multilevel data array. Shape of this data depends on the length of the original
        data and the provided level_size parameter
    """
    binning = _inspect_binning(binning)
    size = int(level_size)
    if size < 2:
        raise ValueError("level_size must be greater than 1")
    #determine multitau level (number of decades)
    level = 0
    while size * 2**(level) < data.shape[-1]:
        level +=1
    
    shape = (level+1,) + data.shape[:-1] + (size,)
    
    out = np.empty(shape = shape, dtype = data.dtype)
    
    for l in range(level+1):
        sliced = data[...,0:size]
        n_sliced = sliced.shape[-1]
        out[l,...,0:n_sliced] = sliced
        
        #missing data... fill nans
        if n_sliced != size:
            out[l,...,n_sliced:] = np.nan
        if l != level:
            if binning == BINNING_MEAN:
                data = convolve(data)[...,::2]
            elif binning == BINNING_FIRST:
                data = data[...,::2]
            else:
                data = choose_data(data, axis = -1)
    return out

def merge_multilevel(data, mode = "full"):
    """Merges multilevel data (data as returned by the :func:`multilevel` function)
    
    Parameters
    ----------
    data : ndarray
        data as returned by :func:`multilevel` 
    mode : str, optional
        Either 'full' or 'half'. Defines how data from the zero-th level of the
        multi-level data is treated. Take all data (full) or only second half
       
    Returns
    -------
    x, y : ndarray, ndarray
        Time, log-spaced data arrays
    """
    if mode == "full":
        x0 = np.arange(data.shape[-1])
        y0 = data[0,...]   
        x, y = merge_multilevel(data[1:], mode = "half")
        return np.concatenate((x0,x*2),axis = -1), np.concatenate((y0,y),axis = -1)
    elif mode == "half":
        n = data.shape[-1]
        if n%2 != 0:
            raise ValueError("Wrong input data shape. Must be even length.")
        
        out_size = n//2 * len(data)
        
        x_slow = np.empty((len(data),n), IDTYPE)
        for i,x in enumerate(x_slow):
            x[:] = np.arange(n)*2**(i)
        x = x_slow[:,n//2:].reshape((out_size,))
        data = data[...,n//2:]
        axes = [i for i in range(data.ndim)]
        axes.insert(-1,axes.pop(0))
        data = data.transpose(axes)
        y = data.reshape(data.shape[:-2] + (out_size,))
        return x,y   
    else:
        raise ValueError("Unknown merging mode")
                 
def log_average(data, size = 8):
    """Performs log average of normalized linear-spaced data.
    
    You must first normalize with :func:`.core.normalize` before averaging!
    
    Parameters
    ----------
    data : array
        Input array of linear-spaced data
    size : int
        Sampling size. Number of data points per each doubling of time.
        Any positive number is valid.
        
    Returns
    -------
    x, y : ndarray, ndarray
        Time and log-spaced data arrays.
    """
    size = int(size)
    if size < 1:
        raise ValueError("`size` must be greater than 0")
    print1("Log-averaging...")
    print2("   * size : {}".format(size))
    ldata = multilevel(data, size*2)
    return merge_multilevel(ldata)

def log_average_count(count, size = 8):
    """Returns effective number of measurements in data calculated
    from linear time-space count data using :func:`log_average`
    
    This function can be used to estimate the effective number of realizations
    of each data point in multilevel data for error estimation.
    
    Parameters
    ----------
    count : ndarray
        Count data, as returned by the `ccorr` or `acorr` functions
    size : int
        Sampling size that was used in :func:`log_average`
        
    Returns
    -------
    x, y : ndarray, ndarray
        Time and log-spaced effetive count arrays.
    """
    data = multilevel_count(count, size * 2, binning = True)
    x, y = merge_multilevel(data, mode = "full")
    return x, y

def _period_from_data(lin, multi):
    nfast = lin.shape[-1]
    nslow = multi.shape[-1]
    period = nfast // nslow
    if nfast % nslow != 0:
        raise ValueError("Fast and slow data are not compatible")
    return period

def log_merge(lin,multi, binning = BINNING_MEAN):
    """Merges normalized multi-tau data.
    
    You must first normalize with :func:`normalize_multi` before merging!
    This function performs a multilevel split on the fast (linear) data and
    merges that with the multilevel slow data into a continuous log-spaced
    data.
    
    Parameters
    ----------
    lin : ndarray
        Linear data
    multi : ndarray
        Multilevel data
    binning : int
        Binning mode used for multilevel calculation of the linear part of the data.
        either BINNING_FIRST or BINNING_MEAN (default).
        
    Returns
    -------
    x, y : ndarray, ndarray
        Time and log-spaced data arrays.
    """
    binning = _inspect_binning(binning)
    period = _period_from_data(lin, multi)
    nslow = multi.shape[-1]

    ldata = multilevel(lin, nslow, binning)
    xfast, cfast = merge_multilevel(ldata, mode = "full")
    xslow, cslow = merge_multilevel(multi, mode = "half")
    t = np.concatenate((xfast, 2 * period * xslow), axis = -1)
    cc = np.concatenate((cfast, cslow), axis = -1)
    return t, cc

def log_merge_count(lin_count,multi_count, binning = BINNING_MEAN):
    """Merges multi-tau count data. This function cab be used to obtain
    effective count data of results merged by :func:`log_merge`. You must
    first call equivalent count function, like :func:`ccorr_multi_count`.
    
    
    Parameters
    ----------
    lin_count : ndarray
        Linear data count
    multi_count : ndarray
        Multilevel data count
    binning : int
        Binning mode used in :func:`log_merge`
        
    Returns
    -------
    x, y : ndarray, ndarray
        Time and log-spaced count arrays.
    """
    period = _period_from_data(lin_count, multi_count)
    nslow = multi_count.shape[-1]

    ldata = multilevel_count(lin_count, nslow, binning = binning)
    xfast, cfast = merge_multilevel(ldata, mode = "full")
    xslow, cslow = merge_multilevel(multi_count, mode = "half")
    t = np.concatenate((xfast, 2 * period * xslow), axis = -1)
    cc = np.concatenate((cfast, cslow), axis = -1)
    return t, cc    
 

def t_multilevel(shape, period = 1):
    """Returns broadcasteable time array of multilevel data with a given shape
    """
    shape = np.array(shape)
    shape[1:-1] = 1 #these are for broadcasting only, so set to one
    out = np.empty(shape = shape, dtype = int)
    x = np.arange(shape[-1]) * period*2
    for o in out:
        o[...,:] = x
        x *= 2
    return out
        
def normalize_multi(data, background = None, variance = None, norm = None,  mode = "corr", 
              scale = False, mask = None):
    """A multitau version of :func:`.core.normalize`.
    
    Performs normalization of data returned by :func:`ccorr_multi`,
     :func:`acorr_multi`,:func:`iccorr_multi`, or :func:`iacorr_multi` function.
    
    See documentation of :func:`.core.normalize`.
    """

    print1("Normalizing...")
    lin, multi = data
    
    #determine what kind of data is there ('diff' or 'corr')
    method = _method_from_data(lin)
    scale = _inspect_scale(scale)
    mode = _inspect_mode(mode)
    if isinstance(norm,str):
        norm = norm_from_string(norm)
    #determine default normalization if not specified by user
    norm = _default_norm_from_data(lin, method, norm)
    
    print2("   * background : {}".format(background is not None))
    print2("   * variance   : {}".format(variance is not None))
    print2("   * norm       : {}".format(norm))
    print2("   * scale      : {}".format(scale))
    print2("   * mode       : {}".format(mode))
    print2("   * mask       : {}".format(mask is not None))
    
    level = disable_prints()
    
    
    if (norm & NORM_WEIGHTED == NORM_WEIGHTED):
    
        norm_comp = (norm & (NORM_COMPENSATED | NORM_SUBTRACTED))|NORM_STRUCTURED
        norm_base = (norm & (NORM_COMPENSATED | NORM_SUBTRACTED)) |NORM_STANDARD     
        
        lin_comp = normalize(lin, background = background, variance = variance, norm = norm_comp, mode = mode,
                  scale = scale, mask = mask)
        multi_comp = normalize(multi, background = background, variance = variance, norm = norm_comp, mode = mode,
                  scale = scale, mask = mask) 
        
        lin_base = normalize(lin, background = background, variance = variance, norm = norm_base, mode = mode,
                  scale = scale, mask = mask)
        multi_base = normalize(multi, background = background, variance = variance, norm = norm_base, mode = mode,
                  scale = scale, mask = mask)   
  
        _scale_factor = np.array(1.,FDTYPE) if scale == True else scale_factor(variance,mask)

        #get data estimator
        x, y = log_merge(lin_comp, multi_comp)
        
        x_lin = np.arange(lin_comp.shape[-1])
        weight = weight_from_data(y, scale_factor = _scale_factor, mode = mode)
        weight = log_interpolate(x_lin,x,weight)
        lin = weighted_sum(lin_comp,lin_base,weight)
        
        period = _period_from_data(lin_comp, multi_comp)
        x_multi = t_multilevel(multi_comp.shape, period = period)
        weight = weight_from_data(y, scale_factor = _scale_factor, mode = mode,)
        weight = log_interpolate(x_multi,x,weight)
        multi = weighted_sum(multi_comp,multi_base, weight)        

    else:
        lin = normalize(lin, background = background, variance = variance, norm = norm, mode = mode,
                  scale = scale, mask = mask)
        multi = normalize(multi, background = background, variance = variance, norm = norm, mode = mode,
                  scale = scale, mask = mask)  
    enable_prints(level)        
    return lin, multi

#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()