from __future__ import absolute_import, print_function, division


import numpy as np
import numba as nb
from cddm.data_select import sector_indexmap, line_indexmap
from cddm.conf import CDTYPE, FDTYPE, IDTYPE, C,F,I64, I64DTYPE, NUMBA_TARGET
from cddm.print_tools import print_progress, print, print_frame_rate
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

_FIGURES = {}

@nb.jit([(C[:],C[:], F[:])], nopython = True)
def _cross_corr_add_vec(xv,yv, out):
    for j in range(xv.shape[0]):
        x = xv[j]
        y = yv[j]
        #calculate cross product
        tmp = x.real * y.real + x.imag * y.imag
        #add 
        out[j] = out[j] + tmp 

@nb.jit([(C[:],C[:]), (F[:],F[:])], nopython = True)
def _add_vec(x, out):
    for j in range(x.shape[0]):
        out[j] = out[j] + x[j] 
        
@nb.jit([(C[:],C[:],F[:])], nopython = True)
def _add_squaresum_vec(x, y, out):
    for j in range(x.shape[0]):
        xx = x[j]
        yy = y[j]
        tmp = xx.real * xx.real + xx.imag* xx.imag
        tmp = tmp + yy.real * yy.real + yy.imag * yy.imag
        out[j] =  out[j] + tmp
        
@nb.jit([(C[:],C[:],F[:])], nopython = True)
def _add_stats_vec(x, out1, out2):
    for j in range(x.shape[0]):
        out1[j] = out1[j] + x[j]
        out2[j] = out2[j] + x[j].real * x[j].real  + x[j].imag * x[j].imag

@nb.guvectorize([(C[:,:],C[:],F[:])],"(m,n)->(n),(n)", target = NUMBA_TARGET)
def _calc_stats_vec(f,out1, out2):
    for i in range(f.shape[0]):
        _add_stats_vec(f[i],out1,out2)

 
@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET)
def _cross_corr_vec(f1,f2,t1,t2,dummy,out):
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
            #else just skip calculation
 
#@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],C[:,:],C[:,:],C[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k),(m,k)", target = "parallel")
#def _cross_mean_vec(f1,f2,t1,t2,dummy,out1,out2):
#    for i in range(f1.shape[0]):
#        for j in range(f2.shape[0]):
#            m=abs(t2[j]-t1[i])
#            if m < out1.shape[0]:
#                _add_vec(f1[i], out1[m])
#                _add_vec(f2[j], out2[m])

@nb.guvectorize([(C[:,:],I64[:],I64[:],C[:,:],C[:,:])],"(l,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET)
def _cross_mean_vec(f,t1,t2,dummy,out):
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _add_vec(f[i], out[m])
           
            
@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:],C[:,:],C[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k),(m,k),(m,k)", target = NUMBA_TARGET)
def _cross_corr_mean_vec(f1,f2,t1,t2,dummy,out,out1,out2):
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
                _add_vec(f1[i], out1[m])
                _add_vec(f2[j], out2[m])

@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:],C[:,:],C[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k),(m,k),(m,k),(m,k)", target = NUMBA_TARGET)
def _cross_corr_mean_square_vec(f1,f2,t1,t2,dummy,out,out1,out2, out3):
    f1.shape
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
                _add_vec(f1[i], out1[m])
                _add_vec(f2[j], out2[m])
                _add_squaresum_vec(f1[i],f2[j], out3[m])
                
@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:],F[:,:],F[:,:],C[:,:],C[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(l,k),(n,k),(m,k)->(m,k),(m,k),(m,k),(m,k)", target = NUMBA_TARGET)
def _cross_corr_mean_square2_vec(f1,f2,t1,t2,f1s,f2s,dummy,out,out1,out2, out3):
    
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
                _add_vec(f1[i], out1[m])
                _add_vec(f2[j], out2[m])
                _add_vec(f2s[j], out3[m])
                _add_vec(f1s[i], out3[m])

@nb.guvectorize([(C[:,:],I64[:],F[:,:],F[:,:])],"(l,k),(l),(m,k)->(m,k)", target = NUMBA_TARGET)
def _auto_corr_vec(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f[j],f[i], out[m])
            #else just skip calculation

@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET)
def _cross_corr_vec2(f1,f2,t1,t2,dummy,out):
    for i in range(f1.shape[0]):
        for j in range(i,f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
            #else just skip calculation

                
@nb.jit([(C[:],C[:], F[:])], nopython = True)
def _cross_diff_add_vec(xv,yv, out):
    for j in range(xv.shape[0]):
        x = xv[j]
        y = yv[j]
        #calculate cross product
        tmp = x-y
        d = tmp.real*tmp.real + tmp.imag*tmp.imag
        #add 
        out[j] = out[j] + d

@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET)
def _cross_diff_vec(f1,f2,t1,t2,dummy,out):
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_diff_add_vec(f2[j],f1[i], out[m])
            #else just skip calculation

@nb.guvectorize([(C[:,:],I64[:],F[:,:],F[:,:])],"(l,k),(l),(m,k)->(m,k)", target = NUMBA_TARGET)
def _auto_diff_vec(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i,f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                _cross_diff_add_vec(f[j],f[i], out[m])
            #else just skip calculation

def _set_out_count(shape, out, n ):
    if out is None:
        out_shape = list(shape)
        if n is not None:
            out_shape[-2] = int(n)
        else:
            n = out_shape[-2]
        out = np.zeros(out_shape, dtype = FDTYPE)
        n = np.zeros((n,), IDTYPE)
    else:
        out, n = out
        out = _transpose_data(out)
        if out.ndim == 1:
            out = out[:,None]   
    return out, n

def _set_out_mean_count(shape, out, n ):
    if out is None:
        out_shape = list(shape)
        if n is not None:
            out_shape[-2] = int(n)
        else:
            n = out_shape[-2]
        out = np.zeros(out_shape, dtype = FDTYPE)
        m1 = np.zeros(out_shape, dtype = CDTYPE)
        m2 = np.zeros(out_shape, dtype = CDTYPE)
        n = np.zeros((n,), IDTYPE)
    else:
        (out, m1, m2, n) = out
        out = _transpose_data(out)
        m1 = _transpose_data(m1)
        m2 = _transpose_data(m2)
        if out.ndim == 1:
            out = out[:,None]   
            m1 = m1[:,None] 
            m2 = m2[:,None] 
    return out, m1, m2, n

def _set_out_mean_square_count(shape, out, n ):
    if out is None:
        out_shape = list(shape)
        if n is not None:
            out_shape[-2] = int(n)
        else:
            n = out_shape[-2]
        out = np.zeros(out_shape, dtype = FDTYPE)
        m1 = np.zeros(out_shape, dtype = CDTYPE)
        m2 = np.zeros(out_shape, dtype = CDTYPE)
        sq = np.zeros(out_shape, dtype = FDTYPE)
        n = np.zeros((n,), IDTYPE)
    else:
        (out, m1, m2, sq, n) = out
        out = _transpose_data(out)
        m1 = _transpose_data(m1)
        m2 = _transpose_data(m2)
        sq = _transpose_data(sq)
        if out.ndim == 1:
            out = out[:,None]   
            m1 = m1[:,None] 
            m2 = m2[:,None] 
            sq = sq[:,None]
    return out, m1, m2, sq, n

#def _set_mean_count(shape, out, n ):
#    if out is None:
#        out_shape = list(shape)
#        if n is not None:
#            out_shape[-2] = int(n)
#        else:
#            n = out_shape[-2]
#        out1 = np.zeros(out_shape, dtype = CDTYPE)
#        out2 = np.zeros(out_shape, dtype = CDTYPE)
#        n = np.zeros((n,), IDTYPE)
#    else:
#        out1, out2, n = out
#        out1 = _transpose_data(out1)
#        out2 = _transpose_data(out2)
#        if out1.ndim == 1:
#            out1 = out1[:,None]  
#        if out2.ndim == 1:
#            out2 = out2[:,None]  
#    return out1,out2, n

def _set_mean_count(shape, out, n ):
    if out is None:
        out_shape = list(shape)
        if n is not None:
            out_shape[-2] = int(n)
        else:
            n = out_shape[-2]
        out = np.zeros(out_shape, dtype = CDTYPE)
        n = np.zeros((n,), IDTYPE)
    else:
        out, n = out
        out = _transpose_data(out)
        if out.ndim == 1:
            out = out[:,None]  
    return out, n

def _set_mean(shape, out, n ):
    if out is None:
        out_shape = list(shape)
        if n is not None:
            out_shape[-2] = int(n)
        out = np.zeros(out_shape, dtype = CDTYPE)
    else:
        out = _transpose_data(out)
        if out.ndim == 1:
            out = out[:,None]  
    return out


def _set_data(f, axis):
    f = np.asarray(f)
    ndim = f.ndim
    
    if ndim == 1:
        f = f[:,None]
    elif ndim > 1:    
        f = np.swapaxes(f,axis,-2)
        
    return f, ndim

def _auto_compute(func, data, t, axis = 0, out = None, n = None):
    print("Computing...")
    #f, ndim, axes = _set_data(data,axis) 
    
    #swap axes or add axes id ndim = 1 so that computation is performed over the -2 axis.
    f, ndim  = _set_data(data,axis) 
    out, count = _set_out_count(f.shape, out, n)          
    if t is None:
        t = np.arange(f.shape[-2], dtype = I64DTYPE)
        
    out = func(f,t,out,out)
    if ndim == 1:
        out = out[:,0]
    _add_count_auto(t,count)
    out = _transpose_data(out)
    return out, count

def _cross_compute(func,f1,f2,t1,t2,axis = 0,out = None, n = None):
    print("Computing...")
    #f1,ndim, axes = _set_data(f1,axis)
    #f2, ndim, axes = _set_data(f2,axis) 
    f1,ndim = _set_data(f1,axis)
    f2, ndim = _set_data(f2,axis) 
    out, count = _set_out_count(f1.shape, out, n) 
    if t1 is None:
        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1   
        
    out = func(f1,f2,t1,t2,out,out)
    if ndim == 1:
        out = out[:,0]
    _add_count_cross(t1,t2,count) 
    out = _transpose_data(out)
    return out, count   

def _cross_compute_mean(func,f1,f2,t1,t2,axis = 0,out = None, n = None):
    print("Computing...")
    #f1,ndim, axes = _set_data(f1,axis)
    #f2, ndim, axes = _set_data(f2,axis) 
    f1,ndim = _set_data(f1,axis)
    f2, ndim = _set_data(f2,axis) 
    out, m1, m2, count = _set_out_mean_count(f1.shape, out, n) 
    
    if t1 is None:
        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1   
        
    out, m1, m2 = func(f1,f2,t1,t2,out,out,m1,m2)
    
    if ndim == 1:
        out = out[:,0]
        m1 = m1[:,0]
        m2 = m2[:,0]
        
    _add_count_cross(t1,t2,count) 
    out = _transpose_data(out)
    m1 = _transpose_data(m1)
    m2 = _transpose_data(m2)
    return (out, m1,m2, count)   

def _cross_compute_mean_square(func,f1,f2,t1,t2,axis = 0,out = None, n = None):
    print("Computing...")
    #f1,ndim, axes = _set_data(f1,axis)
    #f2, ndim, axes = _set_data(f2,axis) 
    f1,ndim = _set_data(f1,axis)
    f2, ndim = _set_data(f2,axis) 
    out, m1, m2, sq, count = _set_out_mean_square_count(f1.shape, out, n) 
    
    
    
    if t1 is None:
        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1   
        
    out, m1, m2, sq = func(f1,f2,t1,t2,out,out,m1,m2,sq)
    
    if ndim == 1:
        out = out[:,0]
        m1 = m1[:,0]
        m2 = m2[:,0]
        sq = sq[:,0]
        
    _add_count_cross(t1,t2,count) 
    out = _transpose_data(out)
    m1 = _transpose_data(m1)
    m2 = _transpose_data(m2)
    sq = _transpose_data(sq)
    return (out, m1,m2, sq, count)  

def _cross_compute_mean_square2(func,f1,f2,t1,t2,f1s,f2s,axis = 0,out = None, n = None):
    print("Computing...")
    #f1,ndim, axes = _set_data(f1,axis)
    #f2, ndim, axes = _set_data(f2,axis) 
    f1,ndim = _set_data(f1,axis)
    f2, ndim = _set_data(f2,axis) 
    f1s,ndim = _set_data(f1s,axis)
    f2s, ndim = _set_data(f2s,axis) 
    
    out, m1, m2, sq, count = _set_out_mean_square_count(f1.shape, out, n) 
    
    if t1 is None:
        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1   
        
    out, m1, m2, sq = func(f1,f2,t1,t2,f1s,f2s,out,out,m1,m2,sq)
    
    if ndim == 1:
        out = out[:,0]
        m1 = m1[:,0]
        m2 = m2[:,0]
        sq = sq[:,0]
        
    _add_count_cross(t1,t2,count) 
    out = _transpose_data(out)
    m1 = _transpose_data(m1)
    m2 = _transpose_data(m2)
    sq = _transpose_data(sq)
    return (out, m1,m2, sq, count) 


#def _cross_mean(func,f1,f2,t1,t2,axis = 0,out = None, n = None):
#    f1, ndim = _set_data(f1,axis)
#    f2, ndim = _set_data(f2,axis) 
#    out1, out2, count = _set_mean_count(f1.shape, out, n) 
#    if t1 is None:
#        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
#    if t2 is None:
#        t2 = t1   
#        
#    out = func(f1,f2,t1,t2,out1,out1, out2)
#    if ndim == 1:
#        out1 = out1[:,0]
#        out2 = out2[:,0]
#    _add_count_cross(t1,t2,count) 
#    out = _transpose_data(out)
#    return out, count  

def _cross_mean(func,f,t1,t2,axis = 0,out = None, n = None):
    f, ndim = _set_data(f,axis)
    out, count = _set_mean_count(f.shape, out, n) 
    if t1 is None:
        t1 = np.arange(f.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1     
    out = func(f,t1,t2,out,out)
    if ndim == 1:
        out = out[:,0]
    _add_count_cross(t1,t2,count) 
    out = _transpose_data(out)
    return out, count  

def _cross_compute2(func,f1,f2,t1,t2,axis = 0,out = None, n = None):
    print("Computing...")
    #f1,ndim, axes = _set_data(f1,axis)
    #f2, ndim, axes = _set_data(f2,axis) 
    f1,ndim = _set_data(f1,axis)
    f2, ndim = _set_data(f2,axis) 
    out, count = _set_out_count(f1.shape, out, n) 
    if t1 is None:
        t1 = np.arange(f1.shape[-2], dtype = I64DTYPE)
    if t2 is None:
        t2 = t1   
        
    out = func(f1,f2,t1,t2,out,out)
    if ndim == 1:
        out = out[:,0]
    _add_count_cross2(t1,t2,count) 
    out = _transpose_data(out)
    return out, count  

def bin_data(data, axis = 0):
    #f, ndim, axes = _set_data(data,axis)
    f, ndim  = _set_data(data,axis)
    f = (f[...,::2,:] + f[...,1::2,:])*0.5
    if ndim == 1:
        #if original data was one dimensional.. make it onedimensional again.
        return f[...,0]
    else:
        #return np.transpose(f,axes)
        return np.swapaxes(f,-2,axis)
  
    
@nb.guvectorize([(C[:],C[:],C[:])],"(m),(m)->(m)", target = NUMBA_TARGET)    
def _random_select(data1,data2, out):
    for i in range(out.shape[0]):
        r = np.random.rand()
        if r >= 0.5:
            out[i] = data1[i]
        else:
            out[i] = data2[i]        

def random_select_data(data, axis = 0):
    f, ndim = _set_data(data,axis)
    f = _random_select(f[...,::2,:],f[...,1::2,:])
    if ndim == 1:
        return f[...,0]
    else:
        return np.swapaxes(f,-2,axis)

def slice_data(data, axis = 0):
    f, ndim = _set_data(data,axis)
    f = f[...,::2,:]
    if ndim == 1:
        return f[...,0]
    else:
        return np.swapaxes(f,-2,axis)

    
def subtract_background(data, axis=0, bg = None, return_bg = False, out = None):
    """Subtracts background frame from a given data array. 
    
    This function can be used to subtract user defined background data, or to
    compute and subtract background data.
    """
    if bg is None:
       bg = np.mean(data, axis = axis)
    data = np.subtract(data,np.expand_dims(bg,axis),out)
    if return_bg == False:
        return data
    else:
        return data, bg

def acorr(f,t = None, axis = 0, n = None, out = None):
    """Computes autocorrelation of the input signal of regular or irregular 
    time - spaced data.
    
    If data has ndim > 1, autocorrelation is performed over the axis defined by 
    the axis parameter. If out is specified the arrays must be zero-initiated.
    
    Parameters
    ----------
    f : array-like
       A complex ND array 
    t : array-like, optional
       Array of integers defining frame times. If not provided, regular time-spaced
       data is assumed.
    out : (ndarray, ndarray), optional
       Tuple of output arrays (corr, count). Both must be zero-initiated by user.
       Calculated results are added to these arrays. 
    n : int, optional
       If provided, determines the length of the output. Note that 'out' parameter
       takes precedence over 'n'.
       
    Returns
    -------
    (corr, count) : (ndarray, ndarray)
       Computed autocorrelation.
        
    Examples
    --------
    
    Say we have two datasets f1 and f2. To compute autocorrelation of both 
    datasets and add these functions together:
    
    >>> data = acorr(f1, n = 64)
    
    Now we can set the 'out' parameter, and the results of the next dataset 'f2'
    are added to results of dataset 'f1':
    
    >>> data = acorr(f2, out = data)
    
    Note that the parameter 'n' = 64 is automatically determined here, based on the
    provide 'out' arrays.  
    """
    return _auto_compute(_auto_corr_vec, f,t,axis = axis ,out = out, n = n)


def adiff(f,t = None, axis = 0,  out = None, n = None):
    return _auto_compute(_auto_diff_vec, f,t,axis = axis ,out = out, n = n)

def ccorr(f1,f2,t1 = None,t2 = None, f1s = None, f2s = None,axis = 0, out = None, n = None, norm = 0 , stats = False):
    if norm == 1:
        out = _cross_compute_mean(_cross_corr_mean_vec,f1,f2,t1,t2,axis = axis ,out = out, n = n)
    elif norm ==2:
        if f1s is None:
            f1s = f1.real * f1.real + f1.imag*f1.imag
        if f2s is None:
            f2s = f2.real * f2.real + f2.imag*f2.imag
        out = _cross_compute_mean_square2(_cross_corr_mean_square2_vec,f1,f2,t1,t2,f1s,f2s,axis = axis ,out = out, n = n)
    elif norm == 0:
        out = _cross_compute(_cross_corr_vec,f1,f2,t1,t2,axis = axis ,out = out, n = n)
    else:
        raise ValueError("Unknown normalization type '{}'".format(norm))
    if stats == True:
        #calculate video statistics (mean and variance) as well and return them
        return out, (f1.mean(axis), f2.mean(axis)), (f1.var(axis), f2.var(axis))
    else:
        return out

#def ccorr(f1,f2,t1 = None,t2 = None, axis = 0, out = None, n = None) :
#    return _cross_compute(_cross_corr_vec,f1,f2,t1,t2,axis = axis ,out = out, n = n)

def cmean(f1,t1 = None, t2 = None, axis = 0, out = None, n = None):
    return _cross_mean(_cross_mean_vec,f1,t1,t2,axis = axis ,out = out, n = n)

def ccorr2(f1,f2,t1 = None,t2 = None, axis = 0, out = None, n = None) :
    return _cross_compute2(_cross_corr_vec2,f1,f2,t1,t2,axis = axis ,out = out, n = n)

def cdiff(f1,f2,t1 = None,t2 = None, axis = 0, out = None, n = None, norm = 0, f1s = None, f2s = None):
    return _cross_compute(_cross_diff_vec,f1,f2,t1,t2,axis = axis ,out = out, n = n)

def _auto_compute_multi(f, t, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None, correlate = True):
    if correlate == True:
        func = acorr
        if binning == True:
            _bin = bin_data
        else:
            _bin = slice_data
    else:
        func = adiff
        if binning == True:
            _bin = random_select_data
        else:
            _bin = slice_data        
    assert n > 4
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t)// (2**(n_decades)) >= n:
            n_decades += 1
        nlog = n_decades -1
        print(n_decades)
    nlog = int(nlog)
    
    assert nlog >= 0


    out_fast, count_fast = func(f,t, axis = axis, n = n_fast)
    shape = out_fast.shape[0:-1]
    out_slow = np.zeros((nlog,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE)  
    out_slow = _transpose_data(out_slow) 
    count_slow = np.zeros((nlog, n_slow,),IDTYPE)
    
    for i in range(nlog):
        f = _bin(f,axis)
        t_slow = np.arange(len(t)//(2**(i+1)))
        func(f,t_slow, axis = axis, out = (out_slow[i], count_slow[i]))
    return (out_fast, count_fast), (out_slow, count_slow)

def acorr_multi(f, t, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None):
    return _auto_compute_multi(acorr, f, t, axis = axis, period = period, n = n, 
                         binning = binning,  nlog = nlog)

def adiff_multi(f, t, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None):
    return _auto_compute_multi(adiff, f, t, axis = axis, period = period, n = n, 
                         binning = binning,  nlog = nlog)
    
    
def _cross_compute_multi(f1,f2, t1, t2, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None, correlate = True, calc_norm = False):
    if correlate == True:
        func = ccorr
        if binning == True:
            _bin = bin_data
        else:
            _bin = slice_data
    else:
        func = cdiff
        if binning == True:
            _bin = random_select_data
        else:
            _bin = slice_data   
    
    assert n > 4
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t1)// (2**(n_decades)) >= n:
            n_decades += 1
        nlog = n_decades -1
    nlog = int(nlog)
    assert nlog >= 0
    
    out_fast, count_fast = func(f1,f2,t1,t2, axis = axis, n = n_fast)
    shape = out_fast.shape[0:-1]
    out_slow = np.zeros((nlog,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE)  
    out_slow = _transpose_data(out_slow) 
    count_slow = np.zeros((nlog, n_slow,),IDTYPE)
    
    for i in range(nlog):
        f1 = _bin(f1,axis)
        f2 = _bin(f2,axis)
        t_slow = np.arange(len(t1)//(2**(i+1)))
        func(f1,f2,t_slow,t_slow, axis = axis, out = (out_slow[i], count_slow[i]))
    return (out_fast, count_fast), (out_slow, count_slow)


def _cross_compute_multi(f1,f2, t1, t2, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None, correlate = True, norm = 0):
    if correlate == True:
        func = ccorr
        if binning == True:
            _bin = bin_data
        else:
            _bin = slice_data
    else:
        func = cdiff
        if binning == True:
            _bin = random_select_data
        else:
            _bin = slice_data   
    
    assert n > 4
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t1)// (2**(n_decades)) >= n:
            n_decades += 1
        nlog = n_decades -1
    nlog = int(nlog)
    assert nlog >= 0
    
    f1s = np.abs(f1)**2 if norm == 2 else None
    f2s = np.abs(f2)**2 if norm == 2 else None

        
    
    data_fast = func(f1,f2,t1,t2, f1s = f1s, f2s = f2s, axis = axis, n = n_fast, norm = norm)
    shape = data_fast[0].shape[0:-1]
    
    
    slow_shape = (nlog,) + shape[0:-1] + (n_slow,) + shape[-1:] 
    #out_slow is allocated with a shape with swaped axes so that it becomes C-contiguous in the ccorr calculation function...  
    out_slow = np.zeros(slow_shape, FDTYPE)  
    #We want output shape to be (nlog,) + shape + (n_slow,), so transpose
    out_slow = _transpose_data(out_slow) 
    count_slow = np.zeros((nlog, n_slow,),IDTYPE)
    if norm == 0:
        out = data_fast, (out_slow, count_slow)
    else:
        m1 = np.zeros((nlog,) + shape[0:-1] + (n_slow,) + shape[-1:], CDTYPE) 
        #We want output shape to be (nlog,) + shape + (n_slow,), so transpose
        m1 = _transpose_data(m1)
        m2 = np.zeros((nlog,) + shape[0:-1] + (n_slow,) + shape[-1:], CDTYPE) 
        #We want output shape to be (nlog,) + shape + (n_slow,), so transpose
        m2 = _transpose_data(m2)
        out = data_fast, (out_slow, m1,m2, count_slow)
    if norm == 2:
        sq = np.zeros((nlog,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE) 
        #We want output shape to be (nlog,) + shape + (n_slow,), so transpose
        sq = _transpose_data(sq)
        out = data_fast, (out_slow, m1,m2, sq, count_slow)    

        

    data_slow = out[1]
    
    for i in range(nlog):
        f1 = _bin(f1,axis)
        f2 = _bin(f2,axis)
        if norm == 2:
            f1s = _bin(f1s,axis)
            f2s = _bin(f2s,axis)
        
        t_slow = np.arange(len(t1)//(2**(i+1)))
        
        _out =  tuple((d[i] for d in data_slow))
        
             
        func(f1,f2,t_slow,t_slow, f1s = f1s, f2s = f2s,axis = axis, norm = norm, out = _out)
        
    return out


def cdiff_multi(f1,f2, t1,t2, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None):
    return _cross_compute_multi(f1,f2, t1,t2, axis = axis, period = period, n = n, 
                         binning = binning,  nlog = nlog, correlate = False)
    
def ccorr_multi(f1,f2, t1,t2, axis = 0, period = 1, n = 2**5, 
                         binning = True,  nlog = None, norm = 0, stats = False):
    
    
    out = _cross_compute_multi(f1,f2, t1,t2, axis = axis, period = period, n = n, 
                         binning = binning,  nlog = nlog, correlate = True, norm = norm)

    if stats == True:
        #calculate video statistics (mean and variance) as well and return them
        return out, (f1.mean(axis), f2.mean(axis)), (f1.var(axis), f2.var(axis))
    else:
        return out

@nb.jit(nopython = True) 
def _add_count_cross(t1,t2,n):
    for ii in range(t1.shape[0]):
        for jj in range(t2.shape[0]):
            m = abs(t1[ii] - t2[jj])
            if m < len(n):
                n[m] += 1   
                
@nb.jit(nopython = True) 
def _add_count_cross2(t1,t2,n):
    for ii in range(t1.shape[0]):
        for jj in range(ii,t2.shape[0]):
            m = abs(t1[ii] - t2[jj])
            if m < len(n):
                n[m] += 1   

@nb.jit(nopython = True) 
def _add_count_auto(t,n):
    for ii in range(t.shape[0]):
        for jj in range(ii,t.shape[0]):
            m = abs(t[ii] - t[jj])
            if m < len(n):
                n[m] += 1   

def auto_analyze_iter(data, t , period = 1, level = 4, 
                        chunk_size = 256,  binning = True, method = "corr",
                        auto_background = False, nlog = None, 
                        return_background = False, index = None):
    if method == "corr":
        f1 = acorr
        f2 = ccorr
    elif method == "diff":
        f1 = adiff
        f2 = cdiff
        binning = False
    else:
        raise ValueError("Unknown method '{}'".format(method))
    
    half_chunk_size = chunk_size // 2
    assert chunk_size % 2 == 0
    assert level > 2
    n = 2 ** level
    assert n <= half_chunk_size 
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t)//(chunk_size * 2**(n_decades)) > 0:
            n_decades += 1
#        while len(t)//(2**(n_decades)) >= n:
#            n_decades += 1
    else:
        n_decades = nlog + 1
        assert n_decades >= 1
    print(n_decades)

        
    t_slow = np.arange(len(t))
        
    print("Computing...")
    
    print_progress(0, len(t))
    
    
    
    for i,d in enumerate(data):
        x1 = d if index is None else d[index]
        
        if i == 0:
            
            shape = x1.shape
            
            out_fast = np.zeros(shape[0:-1] + (n_fast,) + shape[-1:], FDTYPE)
            out_fast = _transpose_data(out_fast)
                
            count_fast = np.zeros((n_fast,),IDTYPE)
            
            out_slow = np.zeros((n_decades-1,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE)  
            out_slow = _transpose_data(out_slow)    
        
            count_slow = np.zeros((n_decades-1, n_slow,),IDTYPE)

            
            fdata1 = np.empty((n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:], CDTYPE)

            
        
        _add_data1(i,x1, fdata1, binning)
                
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



            if auto_background == True:
                if ichunk == 0:
                    bg1 = np.mean(fdata1[0,...,fstart1:fstop1,:], axis = -2)
                np.subtract(fdata1[0,...,fstart1:fstop1,:], bg1[...,None,:], fdata1[0,...,fstart1:fstop1,:])

            if return_background == True:
                out_bg1 = np.mean(fdata1[0,...,fstart1:fstop1,:], axis = -2)
                            
            f1(fdata1[0,...,fstart1:fstop1,:],t[istart1:istop1],-2,out = (out_fast, count_fast)) 

            if istart2 >= 0 :
                f2(fdata1[0][...,fstart1:fstop1,:],fdata1[0][...,fstart2:fstop2,:],t[istart1:istop1],t[istart2:istop2],-2,out = (out_fast, count_fast)) 

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
                
                    if auto_background == True:
                        np.subtract(fdata1[j,...,fstart1:fstop1,:], bg1[...,None,:], fdata1[j,...,fstart1:fstop1,:])

                    
                    f1(fdata1[j,...,fstart1:fstop1,:], t_slow[istart1:istop1],-2, out = (out_slow[j-1],count_slow[j-1]))

                    if istart2 >= 0 :
                        f2(fdata1[j,...,fstart1:fstop1,:],fdata1[j,...,fstart2:fstop2,:], t_slow[istart1:istop1], t_slow[istart2:istop2], -2, out = (out_slow[j-1],count_slow[j-1]))
  
                else:
                    break
            
            print_progress(i+1, len(t))
            
            if return_background == True:
                yield ((out_fast,count_fast), (out_slow,count_slow)), out_bg1
            else:
                yield ((out_fast,count_fast), (out_slow,count_slow))


def cross_analyze_iter(data, t1, t2, period = 1, level = 4, 
                        chunk_size = 256,  binning = True, method = "corr",
                        auto_background = False, nlog = None,  norm = 0,
                        stats = False):
    if method == "corr":
        f = ccorr
    elif method == "diff":
        f = cdiff
        #binning = False
    else:
        raise ValueError("Unknown method '{}'".format(method))
    
    half_chunk_size = chunk_size // 2
    assert chunk_size % 2 == 0
    assert level > 2
    n = 2 ** level
    assert n <= half_chunk_size 
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t1)//(chunk_size * 2**(n_decades)) > 0:
            n_decades += 1
    else:
        n_decades = nlog + 1
        assert n_decades >= 1

    assert n > 4
    n_fast = period * n
    n_slow = n
    if nlog is None:
        n_decades = 0
        while len(t1)// (2**(n_decades)) >= n:
            n_decades += 1
        nlog = n_decades -1
    nlog = int(nlog)
    assert nlog >= 0

        
    t_slow = np.arange(len(t1))
        
    print("Computing...")
    
    print_progress(0, len(t1))
    
    
    
    for i,d in enumerate(data):
        

        x1,x2 = d
        
        if i == 0:
            
            shape = x1.shape
            
            out_fast = np.zeros(shape[0:-1] + (n_fast,) + shape[-1:], FDTYPE)
            out_fast = _transpose_data(out_fast)
            if norm > 0:
                m1_fast = np.zeros(shape[0:-1] + (n_fast,) + shape[-1:], CDTYPE)
                m1_fast = _transpose_data(m1_fast)                  
                m2_fast = np.zeros(shape[0:-1] + (n_fast,) + shape[-1:], CDTYPE)
                m2_fast = _transpose_data(m2_fast) 
            if norm == 2:
                sq_fast = np.zeros(shape[0:-1] + (n_fast,) + shape[-1:], FDTYPE)
                sq_fast = _transpose_data(sq_fast) 
                
            count_fast = np.zeros((n_fast,),IDTYPE)
            
            out_slow = np.zeros((n_decades-1,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE)  
            out_slow = _transpose_data(out_slow)    
            
            if norm > 0:
                m1_slow = np.zeros((n_decades-1,) + shape[0:-1] + (n_slow,) + shape[-1:], CDTYPE)  
                m1_slow = _transpose_data(m1_slow)                  
                m2_slow = np.zeros((n_decades-1,) + shape[0:-1] + (n_slow,) + shape[-1:], CDTYPE)  
                m2_slow = _transpose_data(m2_slow) 
            if norm == 2:
                sq_slow = np.zeros((n_decades-1,) + shape[0:-1] + (n_slow,) + shape[-1:], FDTYPE)  
                sq_slow = _transpose_data(sq_slow) 
                
            count_slow = np.zeros((n_decades-1, n_slow,),IDTYPE)

            
            fdata1 = np.empty((n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:], CDTYPE)
            fdata2 = np.empty((n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:], CDTYPE)
                     
            
            if norm == 2:
                #allocate memory for square data
                sq1 = np.empty((n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:], FDTYPE)
                sq2 = np.empty((n_decades,) + shape[0:-1] + (chunk_size,) + shape[-1:], FDTYPE)
            
            if stats == True:
                sum1 = np.zeros(shape, CDTYPE)
                sum2 =  np.zeros(shape, CDTYPE)
                sqsum1 = np.zeros(shape, FDTYPE)
                sqsum2 = np.zeros(shape, FDTYPE)
                out_bg1 = np.empty(shape, CDTYPE)
                out_bg2 = np.empty(shape, CDTYPE)
                out_var1 = np.empty(shape, FDTYPE)
                out_var2 = np.empty(shape, FDTYPE)    
                
        _add_data2(i,x1, x2, fdata1,fdata2, binning)
        if norm == 2:
            _add_data2(i,np.abs(x1)**2, np.abs(x2)**2, sq1,sq2, binning)
                
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



            if auto_background == True:
                if ichunk == 0:
                    bg1 = np.mean(fdata1[0,...,fstart1:fstop1,:], axis = -2)
                    bg2 = np.mean(fdata2[0,...,fstart1:fstop1,:], axis = -2)
                np.subtract(fdata1[0,...,fstart1:fstop1,:], bg1[...,None,:], fdata1[0,...,fstart1:fstop1,:])
                np.subtract(fdata2[0,...,fstart1:fstop1,:], bg2[...,None,:], fdata2[0,...,fstart1:fstop1,:])

            if stats == True:
                #sum1[...]=0.
                #sum2[...] =0.
                #sqsum1[...]=0.
                #sqsum2[...]=0.
                _calc_stats_vec(fdata1[0,...,fstart1:fstop1,:], sum1, sqsum1)
                _calc_stats_vec(fdata2[0,...,fstart1:fstop1,:], sum2, sqsum2)
                
                _divisor =  (ichunk + 1) * half_chunk_size
                
                np.divide(sum1,_divisor  , out_bg1)
                np.divide(sum2, _divisor , out_bg2)
                np.divide(sqsum1, _divisor , out_var1)
                np.divide(sqsum2, _divisor , out_var2)
                np.subtract(out_var1, np.abs(out_bg1)**2, out_var1)
                np.subtract(out_var2, np.abs(out_bg2)**2, out_var2)


                #out_bg1 = np.mean(fdata1[0,...,fstart1:fstop1,:], axis = -2)
                #out_bg2 = np.mean(fdata2[0,...,fstart1:fstop1,:], axis = -2)
                #out_var1 = (np.abs(fdata1[0,...,fstart1:fstop1,:])**2).sum(axis=-2)/half_chunk_size 
                #out_var2 = (np.abs(fdata2[0,...,fstart1:fstop1,:])**2).sum(axis=-2)/half_chunk_size 

                               
            f1s = sq1[0,...,fstart1:fstop1,:] if norm == 2 else None
            f2s = sq2[0,...,fstart1:fstop1,:] if norm == 2 else None
            
            if norm == 0:
                out = (out_fast, count_fast)
            elif norm == 1:
                out = (out_fast, m1_fast, m2_fast, count_fast)
            else:
                out = (out_fast, m1_fast, m2_fast, sq_fast, count_fast)
            
            f(fdata1[0,...,fstart1:fstop1,:],fdata2[0,...,fstart1:fstop1,:],t1[istart1:istop1],t2[istart1:istop1],f1s = f1s, f2s = f2s,axis = -2, norm = norm,out = out) 

            if istart2 >= 0 :
                
                f1s = sq1[0][...,fstart1:fstop1,:] if norm == 2 else None
                f2s = sq2[0][...,fstart2:fstop2,:] if norm == 2 else None
            
                f(fdata1[0][...,fstart1:fstop1,:],fdata2[0][...,fstart2:fstop2,:],t1[istart1:istop1],t2[istart2:istop2],f1s = f1s, f2s = f2s, axis = -2, norm = norm,out = out) 
 
                f1s = sq1[0][...,fstart2:fstop2,:] if norm == 2 else None
                f2s = sq2[0][...,fstart1:fstop1,:] if norm == 2 else None               
                
                f(fdata1[0][...,fstart2:fstop2,:],fdata2[0][...,fstart1:fstop1,:],t1[istart2:istop2],t2[istart1:istop1],f1s = f1s, f2s = f2s,axis = -2, norm = norm,out = out) 

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
                
                    if auto_background == True:
#                        bg1 = np.mean(fdata1[j,...,fstart1:fstop1,:], axis = -2)
#                        bg2 = np.mean(fdata2[j,...,fstart1:fstop1,:], axis = -2)
                        np.subtract(fdata1[j,...,fstart1:fstop1,:], bg1[...,None,:], fdata1[j,...,fstart1:fstop1,:])
                        np.subtract(fdata2[j,...,fstart1:fstop1,:], bg2[...,None,:], fdata2[j,...,fstart1:fstop1,:])
                    
                    f1s = sq1[j,...,fstart1:fstop1,:] if norm == 2 else None
                    f2s = sq2[j,...,fstart1:fstop1,:] if norm == 2 else None                 
                    
                    if norm == 0:
                        out = (out_slow[j-1],count_slow[j-1])
                    elif norm == 1:
                        out = (out_slow[j-1],m1_slow[j-1],m2_slow[j-1],count_slow[j-1])
                    else:
                        out = (out_slow[j-1],m1_slow[j-1],m2_slow[j-1], sq_slow[j-1],count_slow[j-1])
                        
                    f(fdata1[j,...,fstart1:fstop1,:],fdata2[j,...,fstart1:fstop1,:], t_slow[istart1:istop1], t_slow[istart1:istop1],f1s = f1s, f2s = f2s, axis = -2, norm = norm, out = out)

                    if istart2 >= 0 :
                        f1s = sq1[j,...,fstart1:fstop1,:] if norm == 2 else None
                        f2s = sq2[j,...,fstart2:fstop2,:] if norm == 2 else None                          
                        f(fdata1[j,...,fstart1:fstop1,:],fdata2[j,...,fstart2:fstop2,:], t_slow[istart1:istop1], t_slow[istart2:istop2],f1s = f1s, f2s = f2s, axis = -2, norm = norm, out = out)
                        
                        f1s = sq1[j,...,fstart2:fstop2,:] if norm == 2 else None
                        f2s = sq2[j,...,fstart1:fstop1,:] if norm == 2 else None  
                        f(fdata1[j,...,fstart2:fstop2,:],fdata2[j,...,fstart1:fstop1,:], t_slow[istart2:istop2], t_slow[istart1:istop1],f1s = f1s, f2s = f2s, axis = -2, norm = norm, out = out)
 
                else:
                    break
                
            if norm == 0:
                out = (out_fast,count_fast), (out_slow,count_slow)
            elif norm == 1:
                out = (out_fast,m1_fast, m2_fast, count_fast), (out_slow,m1_slow, m2_slow, count_slow)
            else:
                out = (out_fast,m1_fast, m2_fast, sq_fast, count_fast), (out_slow,m1_slow, m2_slow, sq_slow, count_slow)
            print_progress(i+1, len(t1))
            
            if stats == True:
                yield out, (out_bg1, out_bg2), (out_var1, out_var2)
            else:
                yield out





class RawCorrelationViewer(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """
    def __init__(self, data = None, **kw):
        if data is not None:
            self.init(data, **kw)
            
     
    def init(self,data, bg = None, var = None, semilogx = True, normalize = True):
        
        #self.shape = data.shape
        self.data = data
        self.bg = bg
        self.var = var
        
        _dfast, _dslow = data
        
        (self.data_fast,self.count_fast), (self.data_slow,self.count_slow) = (_dfast[0],_dfast[-1]), (_dslow[0],_dslow[-1])
        self.shape = self.data_fast.shape[0:-1]
        
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        
        self.fig.show()
        plt.subplots_adjust(bottom=0.25)
        
        graph_shape = self.shape[0], self.shape[1]
        self._graph_shape = graph_shape
        
        graph = np.zeros(graph_shape)
        
        max_graph_value = max(graph_shape[0]//2+1, graph_shape[1]) 
        graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value
        
        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
        #self.ax2.grid()
        
        cfast_avg = self.data_fast[0,0]
        cslow_avg = self.data_slow[:,0,0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
        
            cfast_avg = _normalize_fast(cfast_avg, self.count_fast )
            cslow_avg = _normalize_slow(cslow_avg, self.count_slow)
        
        t, avg_data = log_merge(cfast_avg,cslow_avg)
        
        
        if semilogx == True:
            self.l, = self.ax1.semilogx(avg_data)
        else:
            self.l, = self.ax1.plot(avg_data)

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,self.shape[1]-1,valinit = 0, valfmt='%i')

        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = 0, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = 5, valfmt='%.2f') 
                                  
        def update(val):
            self.update()
            
        self.kindex.on_changed(update)
        self.phiindex.on_changed(update)
        self.sectorindex.on_changed(update)
        

        
    def update(self):
        k = self.kindex.val
        phi = self.phiindex.val
        sector = self.sectorindex.val

        graph = np.zeros((self.shape[0], self.shape[1]))
        if sector != 0:
            indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, 1)
        else:
            indexmap = line_indexmap(self.shape[0],self.shape[1],phi)

        
        mask = (indexmap == int(k))
        nans = (indexmap == -1)
        graph = indexmap*1.0 # make it float
        graph[mask] = self._max_graph_value +1
        graph[nans] = np.nan

        
        with np.errstate(divide='ignore', invalid='ignore'):
        
            t, avg_data  = _select_data(self.data, self.bg, self.var, int(k), indexmap, normalize = True)
        avg_data = avg_data#/avg_data[0]
        self.l.set_ydata(avg_data)
        self.l.set_xdata(t)
        
        # recompute the ax.dataLim
        self.ax1.relim()

        # update ax.viewLim using the new dataLim
        self.ax1.autoscale_view()
        
        
        avg_graph = np.zeros(self._graph_shape)
        avg_graph[mask] = 1

        
        self.im.set_data(np.fft.fftshift(graph,0))
        #self.im.set_data(graph)

        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        plt.pause(0.1)
        

    def show(self):
        """Shows video."""
        self.fig.show()
           


class CorrelationViewer(object):
    """Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.
    """
    def __init__(self, data = None, **kw):
        if data is not None:
            self.init(data, **kw)
            
     
    def init(self,data, semilogx = True):
        
        self.t, self.data = data
        self.shape = self.data.shape[0:-1]
        
        self.fig, (self.ax1,self.ax2) = plt.subplots(1,2,gridspec_kw = {'width_ratios':[3, 1]})
        
        self.fig.show()
        plt.subplots_adjust(bottom=0.25)
        
        graph_shape = self.shape[0], self.shape[1]
        self._graph_shape = graph_shape
        
        graph = np.zeros(graph_shape)
        
        max_graph_value = max(graph_shape[0]//2+1, graph_shape[1]) 
        graph[0,0] = max_graph_value 
        
        self._max_graph_value = max_graph_value
        
        self.im = self.ax2.imshow(graph, extent=[0,self.shape[1],self.shape[0]//2+1,-self.shape[0]//2-1])
        #self.ax2.grid()
        
#        cfast_avg = self.data_fast[0,0]
#        cslow_avg = self.data_slow[:,0,0]
#        
#        with np.errstate(divide='ignore', invalid='ignore'):
#        
#            cfast_avg = _normalize_fast(cfast_avg, self.count_fast )
#            cslow_avg = _normalize_slow(cslow_avg, self.count_slow)
#        
#        t, avg_data = log_merge(cfast_avg,cslow_avg)
        k_avg, avg_data  = k_select(self.data, 0 , sector = 0, kstep = 1, k = 1)
        
        if semilogx == True:
            self.l, = self.ax1.semilogx(avg_data)
        else:
            self.l, = self.ax1.plot(avg_data)

        self.kax = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.kindex = Slider(self.kax, "k",0,self.shape[1]-1,valinit = 0, valfmt='%i')

        self.phiax = plt.axes([0.1, 0.10, 0.65, 0.03])
        self.phiindex = Slider(self.phiax, "$\phi$",-90,90,valinit = 0, valfmt='%.2f')        

        self.sectorax = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.sectorindex = Slider(self.sectorax, "sector",0,180,valinit = 5, valfmt='%.2f') 
                                  
        def update(val):
            self.update()
            
        self.kindex.on_changed(update)
        self.phiindex.on_changed(update)
        self.sectorindex.on_changed(update)
        

        
    def update(self):
        k = self.kindex.val
        phi = self.phiindex.val
        sector = self.sectorindex.val

        graph = np.zeros((self.shape[0], self.shape[1]))
        if sector != 0:
            indexmap = sector_indexmap(self.shape[0],self.shape[1],phi, sector, 1)
        else:
            indexmap = line_indexmap(self.shape[0],self.shape[1],phi)

        
        mask = (indexmap == int(k))
        nans = (indexmap == -1)
        graph = indexmap*1.0 # make it float
        graph[mask] = self._max_graph_value +1
        graph[nans] = np.nan

        
        with np.errstate(divide='ignore', invalid='ignore'):
            k_avg, avg_data  = k_select(self.data, phi , sector = sector, kstep = 1, k = k)
        avg_data = avg_data#/avg_data[0]
        self.l.set_ydata(avg_data)
        self.l.set_xdata(self.t)
        
        # recompute the ax.dataLim
        self.ax1.relim()

        # update ax.viewLim using the new dataLim
        self.ax1.autoscale_view()
        
        
        avg_graph = np.zeros(self._graph_shape)
        avg_graph[mask] = 1

        
        self.im.set_data(np.fft.fftshift(graph,0))
        #self.im.set_data(graph)

        self.fig.canvas.draw() 
        self.fig.canvas.flush_events()
        plt.pause(0.1)
        

    def show(self):
        """Shows video."""
        self.fig.show()

        
def iccorr_multi(data, t1, t2, period = 40, level = 4, 
                chunk_size = 256,   binning = True, auto_background = False, nlog = None,   norm = 0, stats = False, show = False):
    
    viewer = None
    method = "corr"
    t0 = time.time()
    for i, data in enumerate(cross_analyze_iter(data, t1, t2, period , level , 
                        chunk_size,  binning,  method, auto_background, nlog, norm, stats)):

        if show == True:
            if viewer is None:
                viewer = RawCorrelationViewer()
                if stats == True:
                    viewer.init(data[0], bg = data[1], var = data[2])
                else:
                    viewer.init(data)
                _FIGURES["ccorr_multi"] = viewer
            else:
                viewer.update()
             
    print_frame_rate(len(t1), t0)
    return data

def iacorr_multi(data, t,  period = 1, level = 4, 
                chunk_size = 256,   binning = True, auto_background = False, nlog = None, show = False, index = 0):
    
    viewer = None
    method = "corr"
    t0 = time.time()
    for data in auto_analyze_iter(data, t,  period , level , 
                        chunk_size,  binning,  method, auto_background, nlog, index):
        if show == True:
            if viewer is None:
                viewer = RawCorrelationViewer()
                viewer.init(data)
                _FIGURES["acorr_multi"] = viewer
            else:
                viewer.update()
    print_frame_rate(len(t), t0)
    return data

def icdiff_multi(data, t1, t2, period = 1, level = 4, 
                        chunk_size = 256,  auto_background = False, nlog = None,show = False):
    
    viewer = None
    method = "diff"
    binning = False
    for data in cross_analyze_iter(data, t1, t2, period , level , 
                        chunk_size,  binning,  method, auto_background, nlog):
        if show == True:
            if viewer is None:
                viewer = RawCorrelationViewer()
                viewer.init(data)
                _FIGURES["cdiff_multi"] = viewer
            else:
                viewer.update()
    return data


def iadiff_multi(data, t , period = 1, level = 4, 
                        chunk_size = 256,  auto_background = False, nlog = None,show = False, index = 0):
    
    viewer = None
    method = "diff"
    binning = False
    for data in auto_analyze_iter(data, t, period , level , 
                        chunk_size,  binning,  method, auto_background, nlog, index):
        if show == True:
            if viewer is None:
                viewer = RawCorrelationViewer()
                viewer.init(data)
                _FIGURES["adiff_multi"] = viewer
            else:
                viewer.update()
    return data


def _normalize_fast(data, count, out = None):
    if out is None:
        out = np.empty_like(data)
    np.divide(data, count, out = out)
    return out

def _normalize_slow(data, count, out = None):
    if out is None:
        out = np.empty_like(data)    
    for d,c,o in zip(data,count,out):
        np.divide(d, c, out = o)
    return out

def normalize(data, inplace = False):
    """Normalizes multi-level or single-level data. Data must be data as returned
    by single-level function acorr, ccorr, adiff, cdiff or their multi-level versions"""
    try:
        (d1,c1), (d2,c2) = data
        out_fast = d1 if inplace == True else None
        out_slow = d2 if inplace == True else None
        return _normalize_fast(d1,c1, out = out_fast), _normalize_slow(d2,c2, out = out_slow)
    except ValueError:
        ##not multi data
        (d1,c1) = data
        out_fast = d1 if inplace == True else None
        return _normalize_fast(d1,c1, out = out_fast)
    

@nb.vectorize([F(F,I64,C,C)],target = NUMBA_TARGET)
def _remove_ccorr_background_simple(data, count, bg1, bg2):
    return data/count - (bg1.real * bg2.real + bg1.imag * bg2.imag)

@nb.vectorize([F(F,I64,C,C,C,C)],target = NUMBA_TARGET)
def _remove_ccorr_background_compensated(data, count, bg1, bg2, m1, m2):
    tmp = data
    tmp = tmp - bg1.real * m2.real - bg1.imag * m2.imag
    tmp = tmp - bg2.real * m1.real - bg2.imag * m1.imag
    return  tmp/count + bg1.real * bg2.real + bg1.imag * bg2.imag

#@nb.vectorize([F(F,I64,C,C,C,C,F,F)],target = "parallel")
#def _remove_ccorr_background_square(data, count, bg1, bg2, m1, m2, sq, offset):
#    tmp = - 0.5 * sq 
#    tmp = tmp  +bg1.real * (m1.real) + bg1.imag * ( m1.imag)
#    tmp = tmp  +bg2.real * (m2.real) + bg2.imag * ( m2.imag)
#    tmp = tmp/count
#    
#    return  data + tmp  + offset
#
#@nb.vectorize([F(F,I64,C,C,C,C,F)],target = "parallel")
#def _remove_ccorr_background_square2(data, count, bg1, bg2, m1, m2, sq):
#    tmp = data - 0.5 * sq 
#    tmp = tmp  +bg1.real * (m1.real-m2.real) + bg1.imag * ( m1.imag-m2.imag)
#    tmp = tmp  +bg2.real * (m2.real-m1.real) + bg2.imag * ( m2.imag-m1.imag)
#    tmp = tmp/count
#
#    
#    return  tmp + bg1.real * bg2.real + bg1.imag * bg2.imag

@nb.vectorize([F(F,I64,C,C,C,C,F)],target = NUMBA_TARGET)
def _remove_ccorr_background_square(data, count, bg1, bg2, m1, m2, sq):
    tmp = data - 0.5 * sq 
    tmp = tmp  +bg1.real * (m1.real-m2.real) + bg1.imag * ( m1.imag-m2.imag)
    tmp = tmp  +bg2.real * (m2.real-m1.real) + bg2.imag * ( m2.imag-m1.imag)
    tmp = tmp/count
    
    d = bg1.real * bg1.real + bg1.imag * bg1.imag
    d = d + bg2.real * bg2.real + bg2.imag * bg2.imag
    
    
    return  tmp - 0.5*d+ bg1.real * bg2.real + bg1.imag * bg2.imag#- d.real * d.real - d.imag * d.imag 

#@nb.vectorize([F(F,I64,C,C)],target = "parallel")
#def _remove_cdiff_background_simple(data, count, bg1, bg2):
#    d = bg2 - bg1
#    d2 = d.real * d.real + d.imag * d.imag
#    return data/count - d2
#
#@nb.vectorize([F(F,I64,C,C,C,C)],target = "parallel")
#def _remove_cdiff_background_compensated(data, count, bg1, bg2, m1, m2):
#    d = bg2 - bg1
#    tmp = data
#    tmp = tmp - bg1.real * m2.real - bg1.imag * m2.imag
#    tmp = tmp - bg2.real * m1.real - bg2.imag * m1.imag
#    return  tmp/count + bg1.real * bg2.real + bg1.imag * bg2.imag


def normalize_ccorr(data, background = None, variance = None, out = None):
    """Normalizes multi-level or single-level data. Data must be data as returned
    by single-level function acorr, ccorr, adiff, cdiff or their multi-level versions"""
    
    #first determine if multi data or not...
    multi_data = False
    n = len(data) # data must be tuple arrays, or 2-tuple of tuples
    if n == 2:
        data_fast, data_slow = data
        if isinstance(data_fast, tuple) and isinstance(data_slow, tuple):
            multi_data = True
            
    if multi_data == False:
        count = data[-1]
        data = data[:-1]
        
        #dimensions of correlation data (the first element of the data tuple)
        ndim = data[0].ndim
        print("ndim",ndim, count.ndim)
        
        if variance is None:
            offset = np.asarray(0.)
        else:
            #offset for norm == 2 type of normalization
            offset = np.asarray(0.5*(variance[0] + variance[1]))
        
        if background is None:
            bg1, bg2 = 0. ,0.
        else:
            bg1, bg2 = background
            bg1 = bg1[...,None] #add dimension for broadcasting
            bg2 = bg2[...,None]
        
        #need to add dimensions to count for broadcasting
        #for 2D data, this is equivalent to count = count[...,None,None,:]
        ndiff = ndim - count.ndim
        for i in range(ndiff):
            count = np.expand_dims(count,-2)
            
        if len(data) == 1:
            return _remove_ccorr_background_simple(data[0], count, bg1, bg2, out = out)
        elif len(data) == 3:
            return _remove_ccorr_background_compensated(data[0], count, bg1, bg2, data[1], data[2], out = out)
        else:
            result = _remove_ccorr_background_square(data[0], count, bg1, bg2, data[1], data[2], data[3],  out = out)
            result += offset[..., None]
            return result
    else:
        if out is None:  
            out_fast, out_slow = None, None
        else:
            out_fast, out_slow = out
            
        out_fast = normalize_ccorr(data_fast, background, out = out_fast, variance = variance)
        
        if out_slow is None:
            out_slow = np.empty_like(data_slow[0])
        out_slow = normalize_ccorr(data_slow, background, out = out_slow, variance = variance)
            
        return out_fast, out_slow        
        
             
@nb.guvectorize([(F[:],F[:])],"(m)->(m)", target = NUMBA_TARGET)
def convolve(a, out):
    n = len(out)
    out[0] = a[0] #at the end we do not use this boundary data so it does not matter what we set here.
    out[n-1] = a[n-1]    
    for i in range(1,n-1):
        out[i] = 0.25*(a[i-1]+2*a[i]+a[i+1])


def log_average(data, level = 4):
    n = 2**level
    assert data.shape[-1] % (n) == 0
    
    p = data.shape[-1] // n
    
    shape = (level+1,) + data.shape[:-1] + (p,)
    
    
    out = np.empty(shape = shape, dtype = data.dtype)
    
    for l in range(level+1):
        out[l] = data[...,0:p]
        data = convolve(data)[...,::2]
    return out
    
def merge_slow_data(data):
    n = data.shape[-1]
    assert n%2 == 0
    
    out_size = n//2 * len(data)
    
    x_slow = np.empty((len(data),n), IDTYPE)
    for i,x in enumerate(x_slow):
        x[:] = np.arange(n)*2**(i+1)
    x = x_slow[:,n//2:].reshape((out_size,))
    data = data[...,n//2:]
    axes = [i for i in range(data.ndim)]
    axes.insert(-1,axes.pop(0))
    data = data.transpose(axes)
    y = data.reshape(data.shape[:-2] + (out_size,))
    return x,y

def merge_fast_data(data):
    x0 = np.arange(data.shape[-1])
    y0 = data[0,...]   
    x, y = merge_slow_data(data[1:])
    return np.concatenate((x0,x),axis = -1), np.concatenate((y0,y),axis = -1)
    
def log_merge(cfast, cslow):
    """Merges fast and slow data."""
    nfast = cfast.shape[-1]
    nslow = cslow.shape[-1]
    period = nfast // nslow
    assert nfast % nslow == 0
    level = int(np.log2(period))
    assert 2**level == period
    ldata = log_average(cfast, level)
    xfast, cfast = merge_fast_data(ldata)
    xslow, cslow = merge_slow_data(cslow)
    t = np.concatenate((xfast, period * xslow), axis = -1)
    cc = np.concatenate((cfast, cslow), axis = -1)
    return t, cc
    

def _transpose_data(data):
    return np.swapaxes(data,-2,-1)


@nb.jit(nopython = True)        
def _bin_data(x1,x2,out):
    out[...] = (x1 + x2)
    out[...] = out*0.5
    return out


def _add_data1(i,x, out, binning = True):
    chunk_size = out.shape[-2]
    n_decades = out.shape[0]
    
    index_top = i % chunk_size
        
    out[0,...,index_top,:] = x 
    
    for j in range(1, n_decades):
        if (i+1) % (2**j) == 0:
            index_low = (i//(2**j)) % (chunk_size)
            
            if binning == True:
                x = _bin_data(x, out[j-1,...,index_top-1,:], out[j,...,index_low,:])
            else:
                out[j,...,index_low,:] = x
                out[j,...,index_low,:] = x
            
            index_top = index_low
        else:
            break

     
def _add_data2(i,x1, x2, out1, out2, binning = True):
    chunk_size = out1.shape[-2]
    n_decades = out1.shape[0]
    
    index_top = i % chunk_size
        
    out1[0,...,index_top,:] = x1
    out2[0,...,index_top,:] = x2   
    
    for j in range(1, n_decades):
        if (i+1) % (2**j) == 0:
            index_low = (i//(2**j)) % (chunk_size)
            
            #x1 = np.add(x1, out1[j-1,...,index_top-1,:], out = out1[j,...,index_low,:])
            #x2 = np.add(x2, out2[j-1,...,index_top-1,:], out = out2[j,...,index_low,:])
            if binning == True:
                x1 = _bin_data(x1, out1[j-1,...,index_top-1,:], out1[j,...,index_low,:])
                x2 = _bin_data(x2, out2[j-1,...,index_top-1,:], out2[j,...,index_low,:])  
            else:
                out1[j,...,index_low,:] = x1
                out2[j,...,index_low,:] = x2
            
            index_top = index_low
        else:
            break

def _select_data(data, k, indexmap, normalize = False):
    _f, _s = data
    (cfast,count_fast), (cslow,count_slow) = (_f[0], _f[-1]), (_s[0],_s[-1])
    
    mask = (indexmap == int(round(k)))
    
    cfast_avg = cfast[mask,:].mean(0)
    cslow_avg = cslow[:,mask,:].mean(1)
    
    if normalize == True:
        normalize_ccorr(((cfast_avg, count_fast),(cslow_avg, count_slow)), out = (cfast_avg, cslow_avg))
        #_normalize_fast(cfast_avg, count_fast, cfast_avg)
        #_normalize_slow(cslow_avg, count_slow, cslow_avg)
    
    tx, cc_avg = log_merge(cfast_avg,cslow_avg)
    
    return tx, cc_avg

def _select_data(data, bg, var, k, indexmap, normalize = False):
    
    cfast, cslow = normalize_ccorr(data, bg, var)

    mask = (indexmap == int(round(k)))
    
    
    cfast_avg = cfast[mask,:].mean(0)
    cslow_avg = cslow[:,mask,:].mean(1)
    
    
    tx, cc_avg = log_merge(cfast_avg,cslow_avg)
    
    return tx, cc_avg

def _k_select(data, k, indexmap, kmap):
    mask = (indexmap == int(round(k)))
    ks = kmap[mask]
    if len(ks) > 0:
        #at least one k value exists
        data_avg = data[...,mask,:].mean(-2)
        k_avg = ks.mean()
        return k_avg, data_avg  
    else:
        #no k values
        return None

def _k_map(height, width):
    x,y = np.meshgrid(np.fft.fftfreq(height)*height,range(width), indexing = "ij")  
    x2, y2  = x**2, y**2
    return (x2 + y2)**0.5    

def k_select(data, phi , sector = 5, kstep = 1, k = None):
    """k-selection and k-averaging of correlation data.
    
    This function takes (...,i,j,n) correlation data and performs k-based selection
    and averaging of the data.
    
    Parameters
    ----------
    deta : array_like
        Input correlation data of shape (...,i,j,n)
    phi : float
        Angle between the j axis and the direction of k-vector
    sector : float, optional
        Defines sector angle, k-data is avergaed between phi+sector and phi-sector angles
    kstep : float, optional
        Defines an approximate k step in pixel units
    k : float or list of floats, optional
        If provided, only return at a given k value (and not at all non-zero k values)
        
        
    Returns
    -------
    out : iterator, or tuple
        If k s not defined, this is an iterator that yields a tuple of (k_avg, data_avg)
        of actual (mean) k and averaged data. If k is a list of indices, it returns
         an iterator that yields a tuple of (k_avg, data_avg) for every non-zero data
        if k is an integer, it returns a tuple of (k_avg, data_avg) for a given k-index.
    """
    
    if sector > 0:
        indexmap = sector_indexmap(data.shape[-3], data.shape[-2], phi, sector, kstep)
    else:
        indexmap = line_indexmap(data.shape[-3], data.shape[-2], phi)
    kmap = _k_map(data.shape[-3], data.shape[-2])  

    if k is None:
        kdim = indexmap.max() + 1
        k = np.arange(kdim)
    try:
        iterator = (_k_select(data, kk, indexmap, kmap) for kk in k)
        return tuple((data for data in iterator if data is not None))       
    except TypeError:
        #not iterable
        return _k_select(data, k, indexmap, kmap)


