"""
Low level numba functions
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import numba as nb
from cddm.conf import C,F, I64, NUMBA_TARGET, NUMBA_FASTMATH, NUMBA_CACHE
from cddm.fft import _fft, _ifft
from cddm.decorators import doc_inherit




#Some useful functions

@nb.vectorize([F(C)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def abs2(x):
    """Absolute square of data"""
    return x.real*x.real + x.imag*x.imag

@nb.vectorize([F(F,F),C(C,C)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def mean(a,b):
    """Man value"""
    return 0.5 * (a+b)

@nb.vectorize([F(F,F),C(C,C)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def choose(a,b):
    """Chooses data randomly"""
    r = np.random.rand()
    if r >= 0.5:
        return a
    else:
        return b   
 
@nb.guvectorize([(F[:],F[:])],"(m)->(m)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def convolve(a, out):
    """Convolves input array with kernel [0.25,0.5,0.25]"""
    n = len(out)
    assert n > 2  
    result = a[0]
    for i in range(1,n-1):
        out[i-1] = result 
        result = 0.25*(a[i-1]+2*a[i]+a[i+1])

    out[i] = result
    out[-1] = a[-1]

@nb.guvectorize([(F[:],F[:],F[:],F[:])],"(n),(m),(m)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def interpolate(x_new, x,y,out):
    """Linear interpolation"""
    assert len(x) >= 2
    for i in range(len(x_new)):
        xi = x_new[i]
        for j in range(1,len(x)):
            x0 = x[j-1]
            x1 = x[j]
            if xi <= x1:
                #interpolate or extrapolate backward
                deltay = y[j] - y[j-1]
                deltax = x1 - x0
                out[i] = (xi - x0) * deltay/ deltax +  y[j-1]
                break
        #extrapolate forward
        if xi > x1:
            deltay = y[-1] - y[-2]
            deltax = x1 - x0
            out[i] = (xi - x0) * deltay/deltax +  y[-2]            

@nb.guvectorize([(I64[:],I64[:],F[:],F[:])],"(n), (m),(m)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _log_interpolate(x_new, x,y, out):
    """Linear interpolation in semilogx space."""
    assert len(x) >= 2
    for i in range(len(x_new)):
        xi = x_new[i]
        log = False
        if xi > 1:
            xi = np.log(xi)
            log = True
        for j in range(1,len(x)):
            x0 = x[j-1]
            x1 = x[j]            
            if x0 >= 1 and log == True:
                x0 = np.log(x0)
                x1 = np.log(x1)
            if x_new[i] <= x[j]:
                #interpolate or extrapolate backward
                deltay = y[j] - y[j-1]
                deltax = x1-x0
                out[i] = (xi - x0) * deltay / deltax +  y[j-1]
                break
        #extrapolate forward for data points outside of the domain
        if xi > x1:            
            deltay = y[-1] - y[-2]
            deltax = x1 - x0
            out[i] = (xi - x0) * deltay/deltax +  y[-2]    

def log_interpolate(x_new, x,y, out = None):
    """Linear interpolation in semilogx space."""
    #wrapped to suprres divide by zero warning numba issue #4793
    with np.errstate(divide='ignore'):
        return _log_interpolate(x_new, x,y, out)
    
log_interpolate.__doc__ = _log_interpolate.__doc__

@nb.guvectorize([(F[:],F[:])],"(n)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def median(array, out):
    """Performs median filter."""
    n = len(array)
    assert n > 2
    result_out = array[0]
    out[-1] = array[-1]
    for i in range(1,n-1):
        if array[i] < array[i+1]:
            if array[i] < array[i-1]:
                result = min(array[i+1],array[i-1])
            else:
                result = array[i]
        else:
            if array[i] < array[i-1]:
                result = array[i]
            else:
                result = max(array[i+1],array[i-1])
        out[i-1] = result_out
        result_out = result
    out[i] = result_out
    #out[n-1] = result_out
    #out[0] = out[1]

@nb.vectorize([F(F,F,F)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def weighted_sum(x, y, weight):
    """Performs weighted sum of two data sets, given the weight data.
    Weight must be normalized between 0 and 1. Performs:
    `x * weight + (1.- weight) * y`
    """ 
    return x * weight + (1.- weight) * y
    
@nb.guvectorize([(F[:],F[:])],"(n)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _median_slow(array, out):
    """Performs median filter. slow implementation... for testing"""
    n = len(array)
    assert n > 2
    for i in range(1,n-1):
        median = np.sort(array[i-1:i+2])[1]
        out[i] = median
    out[0] = array[0]
    out[-1] = array[-1]
    #out[0] = out[1]
    #out[n-1] = out[n-2]

@nb.guvectorize([(F[:],F[:])],"(n)->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def decreasing(array, out):
    """Performs decreasing filter. Each next element must be smaller or equal"""
    n = len(array)
    for i in range(n):
        if i == 0:
            out[0] = array[0]
        else:
            if array[i] < out[i-1] or np.isnan(out[i-1]):
                out[i] = array[i]
            else:
                out[i] = out[i-1]
                
                
                
@nb.guvectorize([(F[:],F[:])],"(n)->(n)", target = "cpu", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def increasing(array,out):
    """Performs increasing filter. Each next element must be greater or equal"""
    n = len(array)
    for i in range(1,n):
        if i == 0:
            out[0] = array[0]
        else:
            if array[i] > out[i-1] or np.isnan(out[i-1]):
                out[i] = array[i]
            else:
                out[i] = out[i-1]

#------------------------------------------------
# low level numba-optimized computation functions
#------------------------------------------------

@nb.jit([(C[:],C[:], F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr_add_vec(xv,yv, out):
    for j in range(xv.shape[0]):
        x = xv[j]
        y = yv[j]
        #calculate cross product
        tmp = x.real * y.real + x.imag * y.imag
        #add 
        out[j] = out[j] + tmp 

@nb.jit([(C[:],C[:]), (F[:],F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _add_vec(x, out):
    for j in range(x.shape[0]):
        out[j] = out[j] + x[j] 
        
@nb.jit([(C[:],C[:],F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _add_squaresum_vec(x, y, out):
    for j in range(x.shape[0]):
        xx = x[j]
        yy = y[j]
        tmp = xx.real * xx.real + xx.imag* xx.imag
        tmp = tmp + yy.real * yy.real + yy.imag * yy.imag
        out[j] =  out[j] + tmp
        
@nb.jit([(C[:],C[:],F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _add_stats_vec(x, out1, out2):
    for j in range(x.shape[0]):
        out1[j] = out1[j] + x[j]
        out2[j] = out2[j] + x[j].real * x[j].real  + x[j].imag * x[j].imag

@nb.guvectorize([(C[:,:],C[:],F[:])],"(m,n)->(n),(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _calc_stats_vec(f,out1, out2):
    for i in range(f.shape[0]):
        _add_stats_vec(f[i],out1,out2)

 
@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr_vec(f1,f2,t1,t2,dummy,out):
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f2[j],f1[i], out[m])
                
@nb.guvectorize([(C[:],C[:],I64[:],I64[:],F[:],F[:])],"(m),(n),(m),(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr(x,y,t1,t2,dummy,out):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            #m = abs(j-i)
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                tmp = x[i].real * y[j].real + x[i].imag * y[j].imag
                out[m] += tmp

@nb.jit([(C[:],C[:], F[:])], nopython = True)
def _cross_corr_add(xv,yv, out):
    for j in range(xv.shape[0]):
        x = xv[j]
        y = yv[j]
        #calculate cross product
        tmp = x.real * y.real + x.imag * y.imag
        #add 
        out[0] = out[0] + tmp 

@nb.guvectorize([(C[:],C[:],F[:],F[:])],"(n),(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr_regular(x,y,dummy,out):
    for i in range(out.shape[0]):
        n = x.shape[0] - i
        _cross_corr_add(y[i:],x[0:n], out[i:i+1])
        if i > 0:
            _cross_corr_add(y[0:n],x[i:], out[i:i+1])

@nb.guvectorize([(C[:],F[:],F[:])],"(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_corr_regular(x,dummy,out):
    for i in range(out.shape[0]):
        n = x.shape[0] - i
        _cross_corr_add(x[i:],x[0:n], out[i:i+1])


@nb.guvectorize([(C[:,:],I64[:],F[:,:],F[:,:])],"(l,k),(l),(m,k)->(m,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_corr_vec(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                _cross_corr_add_vec(f[j],f[i], out[m])
            #else just skip calculation
            
@nb.guvectorize([(C[:],I64[:],F[:],F[:])],"(l),(l),(m)->(m)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_corr(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                tmp = f[i].real * f[j].real + f[i].imag * f[j].imag
                out[m] += tmp
              
@nb.jit([(C[:],C[:], F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_diff_add_vec(xv,yv, out):
    for j in range(xv.shape[0]):
        x = xv[j]
        y = yv[j]
        #calculate cross product
        tmp = x-y
        d = tmp.real*tmp.real + tmp.imag*tmp.imag
        #add 
        out[j] = out[j] + d

@nb.guvectorize([(C[:,:],C[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(n,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_diff_vec(f1,f2,t1,t2,dummy,out):
    for i in range(f1.shape[0]):
        for j in range(f2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _cross_diff_add_vec(f2[j],f1[i], out[m])
            #else just skip calculation
                
@nb.guvectorize([(C[:],C[:],I64[:],I64[:],F[:],F[:])],"(m),(n),(m),(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_diff(x,y,t1,t2,dummy,out):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            #m = abs(j-i)
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                tmp = y[j]-x[i]
                d = tmp.real*tmp.real + tmp.imag*tmp.imag
                out[m] += d

@nb.jit([(C[:],C[:], F[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_diff_add(xv,yv, out):
    for j in range(xv.shape[0]):
        tmp = xv[j] - yv[j] 
        d = tmp.real*tmp.real + tmp.imag*tmp.imag
        out[0] += d

@nb.guvectorize([(C[:],C[:],F[:],F[:])],"(n),(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_diff_regular(x,y,dummy,out):
    for i in range(out.shape[0]):
        n = x.shape[0] - i
        _cross_diff_add(y[i:],x[0:n], out[i:i+1])
        if i > 0:
            _cross_diff_add(y[0:n],x[i:], out[i:i+1])

@nb.guvectorize([(C[:,:],I64[:],F[:,:],F[:,:])],"(l,k),(l),(m,k)->(m,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_diff_vec(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i,f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                _cross_diff_add_vec(f[j],f[i], out[m])
            #else just skip calculation
            
@nb.guvectorize([(C[:],F[:],F[:])],"(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_diff_regular(x,dummy,out):
    for i in range(out.shape[0]):
        n = x.shape[0] - i
        _cross_diff_add(x[i:],x[0:n], out[i:i+1])
            
@nb.guvectorize([(C[:],I64[:],F[:],F[:])],"(l),(l),(m)->(m)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_diff(f,t,dummy,out):
    for i in range(f.shape[0]):
        for j in range(i, f.shape[0]):
            m=abs(t[j]-t[i])
            if m < out.shape[0]:
                tmp = f[j] - f[i] 
                d = tmp.real*tmp.real + tmp.imag*tmp.imag
                out[m] += d


@nb.guvectorize([(C[:,:],I64[:],I64[:],C[:,:],C[:,:]),(F[:,:],I64[:],I64[:],F[:,:],F[:,:])],"(l,k),(l),(n),(m,k)->(m,k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_sum_vec(f,t1,t2,dummy,out):
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                _add_vec(f[i], out[m])

@nb.guvectorize([(C[:],I64[:],I64[:],C[:],C[:]),(F[:],I64[:],I64[:],F[:],F[:])],"(l),(l),(n),(m)->(m)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_sum(f,t1,t2,dummy,out):
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            m=abs(t2[j]-t1[i])
            if m < out.shape[0]:
                out[m] += f[i]
                                
@nb.guvectorize([(C[:],C[:],C[:]), (F[:],F[:],F[:])],"(n),(k)->(k)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_sum_regular(x,dummy,out):
    for i in range(out.shape[0]):
        n = x.shape[0] - i
        if i == 0:
            tmp = out[0]
            tmp = tmp*0.
            for j in range(x.shape[0]):
                tmp = tmp + x[j]
            prev = tmp*2
            out[0] += tmp
        if i > 0:
            prev = prev - x[i-1] - x[n]
            out[i] += prev 

@nb.guvectorize([(C[:],C[:],F[:],F[:])],"(n),(n),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr_fft_regular(x, y, dummy, out):
    out_length = len(dummy)
    length = len(x)
    tmp1 = np.empty((length*2), x.dtype)
    tmp1[0:length] = x
    tmp1[length:] = 0.
    tmp2 = np.empty((length*2), y.dtype)
    tmp2[0:length] = y
    tmp2[length:] = 0.  
    x = _fft(tmp1, overwrite_x = True)
    y = _fft(tmp2, overwrite_x = True)
    
    x = x*np.conj(y)

    _out = _ifft(x, overwrite_x = True)

    out[:] += _out[:out_length].real
    out[1:] += _out[-1:-out_length:-1].real
    
@nb.guvectorize([(C[:],F[:],F[:])],"(n),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_corr_fft_regular(x, dummy, out):
    out_length = len(dummy)
    length = len(x)
    tmp = np.empty((length*2), x.dtype)
    tmp[0:length] = x
    tmp[length:] = 0.
 
    x = _fft(tmp, overwrite_x = True)
    x = x*np.conj(x)

    _out = _ifft(x, overwrite_x = True)
    out[:] += _out[:out_length].real    

@nb.jit([(C[:],I64[:],C[:]),(F[:],I64[:],C[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _fill_data(x,t, out):
    for i in range(t.shape[0]):
        m = t[i]
        if m < out.shape[0]:
            out[m] = x[i]

@nb.jit([(I64[:],C[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _fill_ones(t, out):
    for i in range(t.shape[0]):
        m = t[i]
        if m < out.shape[0]:
            out[m] = 1.
         
@nb.guvectorize([(C[:],C[:],I64[:],I64[:],I64[:],F[:],F[:])],"(n),(n),(n),(n),(),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_corr_fft(x, y, t1,t2,length, dummy, out):
    out_length = len(dummy)
    
    tmp1 = np.zeros((length*2), x.dtype)
    _fill_data(x,t1, tmp1)
    #tmp1[list(t1)] = x
    tmp2 = np.zeros((length*2), y.dtype)
    _fill_data(y,t2, tmp2)
    #tmp2[list(t2)] = y 
    x = _fft(tmp1, overwrite_x = True)
    y = _fft(tmp2, overwrite_x = True)
    np.conj(y, out = y)
    np.multiply(x,y, out = x)
    _out = _ifft(x, overwrite_x = True)
    out[:] += _out[:out_length].real
    out[1:] += _out[-1:-out_length:-1].real
    
@nb.guvectorize([(C[:],I64[:],I64[:],F[:],F[:])],"(n),(n),(),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _auto_corr_fft(x, t,length, dummy, out):
    out_length = len(dummy)
    
    tmp = np.zeros((length*2), x.dtype)
    _fill_data(x,t, tmp)
    #tmp[list(t)] = x
    x = _fft(tmp, overwrite_x = True)
    y = np.conj(x)
    np.multiply(x,y, out = x)
    _out = _ifft(x, overwrite_x = True)
    out[:] += _out[:out_length].real

@nb.guvectorize([(C[:],C[:],I64[:],I64[:],C[:],C[:])],"(n),(m),(n),(),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_sum_fft(x, y,t,length, dummy, out):
    out_length = len(dummy)
    tmp1 = np.zeros((length*2), x.dtype)
    _fill_data(x,t, tmp1)
    x = _fft(tmp1, overwrite_x = True)
    np.multiply(x,y, out = x)
    _out = _ifft(x, overwrite_x = True)
    out[:] += _out[:out_length]
    out[1:] += _out[-1:-out_length:-1]


@nb.guvectorize([(F[:],C[:],I64[:],I64[:],F[:],F[:])],"(n),(m),(n),(),(k)->(k)", forceobj=True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _cross_sum_rfft(x, y,t,length, dummy, out):
    out_length = len(dummy)
    tmp1 = np.zeros((length*2), y.dtype)
    _fill_data(x,t, tmp1)
    x = _fft(tmp1, overwrite_x = True)
    np.multiply(x,y, out = x)
    _out = _ifft(x, overwrite_x = True)
    out[:] += _out[:out_length].real
    out[1:] = out[1:] + _out[-1:-out_length:-1].real

#-----------------------------
# occurence count functions

@nb.jit(nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _add_count_cross(t1,t2,n):
    for ii in range(t1.shape[0]):
        for jj in range(t2.shape[0]):
            m = abs(t1[ii] - t2[jj])
            if m < len(n):
                n[m] += 1   
                
# @nb.jit([(I64[:],I64[:],I64[:,:],I64[:,:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def cross_tau_time(t1,t2,tn1,tn2):
#     assert len(t1) == len(t2) 
#     assert tn1.shape == tn2.shape
    
#     count = np.zeros((tn1.shape[0],),tn1.dtype)
    
#     for ii in range(t1.shape[0]):
#         for jj in range(t2.shape[0]):
#             m = abs(t1[ii] - t2[jj])
#             if m < tn1.shape[0]:
#                 i = count[m]
#                 if i < tn1.shape[1]:
#                     tn1[m,i] = t1[ii]   
#                     tn2[m,i] = t2[jj] 
#                     count[m] +=1


@nb.jit([(I64[:],I64[:],I64[:,:],I64[:,:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def cross_tau_times(t1,t2,tpos,tneg):
    assert len(t1) == len(t2) 
    assert tpos.shape == tpos.shape
    
    count_pos = np.zeros((tpos.shape[0],),tpos.dtype)
    count_neg = np.zeros((tneg.shape[0],),tneg.dtype)
    
    for ii in range(t1.shape[0]):
        for jj in range(t2.shape[0]):
            m = t1[ii] - t2[jj]
            if abs(m) < tpos.shape[0]:
                if m > 0:
                    i = count_pos[m]
                    if i < tpos.shape[1]:
                        tpos[m,i] = t1[ii]   
                        count_pos[m] +=1
                else:
                    m = -m
                    i = count_neg[m]
                    if i < tneg.shape[1]:
                        tneg[m,i] = t1[ii]   
                        count_neg[m] +=1

@nb.jit([(I64[:],I64[:,:])],cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def auto_tau_times(t,tpos):

    count_pos = np.zeros((tpos.shape[0],),tpos.dtype)

    for ii in range(t.shape[0]):
        for jj in range(ii,t.shape[0]):
            m = t[jj] - t[ii]
            assert m >= 0
            if m < tpos.shape[0]:
                i = count_pos[m]
                if i < tpos.shape[1]:
                    tpos[m,i] = t[ii]   
                    count_pos[m] +=1


def cross_count_mixed(t1,t2, n, period):
    pos = np.empty((n,len(t1)//period * 2),int)
    pos[...] = -1
    neg = np.empty((n,len(t1)//period * 2),int)
    neg[...] = -1
    cross_tau_times(t1,t2,pos,neg)
    
    count_pos_pos = np.zeros((n, 2*n), int)
    count_neg_neg = np.zeros((n, 2*n), int)
    count_pos_neg = np.zeros((n, 2*n), int)
    
    for i in range(n):
        pmask = pos[i] > 0
        _add_count_cross(pos[i,pmask],pos[i,pmask],count_pos_pos[i])
        nmask = neg[i] > 0
        _add_count_cross(neg[i,nmask],neg[i,nmask],count_neg_neg[i])
        _add_count_cross(pos[i,pmask],neg[i,nmask],count_pos_neg[i])
    return count_pos_pos,count_neg_neg,count_pos_neg
        
def auto_count_mixed(t, n, period):
    pos = np.empty((n,len(t)//period * 2),int)
    pos[...] = -1

    auto_tau_times(t,pos)
    count = np.zeros((n, 2*n), int)

    
    for i in range(n):
        mask = pos[i] > 0
        _add_count_cross(pos[i,mask],pos[i,mask],count[i])

    return count
        
        
    
    


@nb.jit(nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def _add_count_auto(t,n):
    for ii in range(t.shape[0]):
        for jj in range(ii,t.shape[0]):
            m = abs(t[ii] - t[jj])
            if m < len(n):
                n[m] += 1   
    
    
#normalization functions
#-----------------------
    
@nb.vectorize([F(F,I64,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_0(data, count, bg1, bg2):
    return data/count - (bg1.real * bg2.real + bg1.imag * bg2.imag)

@nb.vectorize([F(F,I64,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_cdiff_1(data, count, d):
    return data/count - (d.real * d.real + d.imag*d.imag)

@nb.vectorize([F(F,I64,C,C,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_2(data, count, bg1, bg2, m1, m2):
    tmp = data
    tmp = tmp - bg1.real * m2.real - bg1.imag * m2.imag
    tmp = tmp - bg2.real * m1.real - bg2.imag * m1.imag
    return  tmp/count + bg1.real * bg2.real + bg1.imag * bg2.imag

@nb.vectorize([F(F,I64,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_2b(data, count, m1, m2):
    tmp = m1.real * m2.real + m1.imag * m2.imag
    tmp = tmp/count
    tmp = data -tmp
    return tmp/count

@nb.vectorize([F(F,I64,C,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_cdiff_3(data, count, dm, m1, m2):
    ds = m2 - m1
    tmp = data - 2*(dm.real * ds.real + dm.imag * ds.imag)
    return  tmp/count  + (dm.real * dm.real + dm.imag * dm.imag)

@nb.vectorize([F(F,I64,C,C,F,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_3(data, count, bg1, bg2, sq, m1, m2):
    tmp = data - 0.5 * sq 
    
    tmp = tmp + (m1.real-m2.real)* (bg1.real- bg2.real)
    tmp = tmp + (m1.imag-m2.imag)* (bg1.imag- bg2.imag)

    tmp = tmp/count
    
    d = (bg1.real - bg2.real)
    d2 = d*d
    d = (bg1.imag - bg2.imag)
    d2 = d2 + d*d
    
    return tmp - (0.5 * d2)

@nb.vectorize([F(F,I64,F,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_3b(data, count, sq, m1, m2):

    
    tmp = (m1.real-m2.real)* (m1.real- m2.real)
    tmp = tmp + (m1.imag-m2.imag)* (m1.imag- m2.imag)
    tmp = 0.5*tmp/count
    tmp = data + tmp - 0.5 * sq 

    
    return tmp/count

@nb.vectorize([F(F,I64,C,C,F)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_1(data, count, bg1, bg2, sq):
    tmp = data - 0.5 * sq 
    
    tmp = tmp/count
    
    d = (bg1.real - bg2.real)
    d2 = d*d
    d = (bg1.imag - bg2.imag)
    d2 = d2 + d*d
    
    return tmp + (0.5 * d2)

#because of numba bug, this does not work for np.nan inputs

# @nb.jit([F(F,F)], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def _weight_from_g(g,delta):
#     tmp1 = 2*g
#     tmp2 = g**2 + 1 + 2*delta**2
#     return tmp1/tmp2    

@nb.vectorize([F(C,F)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def weight_from_g(g, delta):
    """Computes weight for weighted normalization from normalized and scaled 
    correlation function"""
    tmp1 = 2*g.real
    g2 = g.real**2 + g.imag**2
    tmp2 = g2 + 1 + delta**2
    return tmp1/tmp2    


@nb.vectorize([F(C,F,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def weight_prime_from_g(g,delta, b1, b2):
    """Computes weight for weighted normalization from normalized and scaled 
    correlation function"""
    # s1 = |b1|^2
    s1 = b1.real * b1.real + b1.imag * b1.imag
    # s2 = |b2|^2
    s2 = b2.real * b2.real + b2.imag * b2.imag
    # r = Re(conj(b2)*b1)
    r = b1.real * b2.real + b1.imag * b2.imag
    #i = Im(conj(b2)*b1)
    i = b2.real * b1.imag - b2.imag * b1.real
    
    d2 = delta**2
    
    g2 = g.real**2 + g.imag**2
    
    tmp1 = 2 * g.real + 2 * r + (s1 + s2) * g.real
    tmp2 = g2 + 1 + d2 + s1 + s2 + (s2 - s1) * delta + 2 * r * g.real + 2 * i * g.imag
    return tmp1/tmp2    

def weight_prime_from_d(d, delta, b1, b2):
    g = 1 - d/2.
    return weight_prime_from_g(g,delta, b1, b2)

def weight_from_d(d, delta):
    g = 1 - d/2.
    return weight_from_g(g, delta)
     

@nb.vectorize([F(F,C,F)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def sigma_weighted(w,g,delta):
    """Computes standard deviation of the weighted normalization."""
    g2 = g.real**2 
    c2= g2 + g.imag**2
    d2 = delta**2
    return (0.5 * (w**2 * (c2 + 1 + d2) - 4 * w * g.real   + 2*g2 - c2 + 1 - d2))**0.5


@nb.vectorize([F(F,C,F,C,C)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def sigma_prime_weighted(w,g,delta, b1, b2):
    g2 = g.real**2
    c2 = g2 + g.imag**2
    d2 = delta**2
    # s1 = |b1|^2
    s1 = b1.real * b1.real + b1.imag * b1.imag
    # s2 = |b2|^2
    s2 = b2.real * b2.real + b2.imag * b2.imag
    # r = Re(conj(b2)*b1)
    r = b1.real * b2.real + b1.imag * b2.imag
    #i = Im(conj(b2)*b1)
    i = b2.real * b1.imag - b2.imag * b1.real
    
    return (0.5 * (w**2 * (c2 + 1 + d2 + s1 + s2 + (s2 - s1) * delta + 2 * r * g.real + 2 * i * g.imag) \
                   - 4 * w * (g.real + r + 0.5 * (s1 + s2) * g.real ) \
                       + 2*g2 - c2 + 1 - d2 + s1 + s2 - (s2 - s1) * delta + 2 * r * g.real - 2 * i * g.imag))**0.5




@nb.jit
def _g(a,index):
    index = abs(index)
    if index > len(a):
        return 0.
    else:
        return a[index]

# @nb.guvectorize([(F[:],F[:],F[:],F[:],I64[:,:],I64[:,:],I64[:,:],F[:])],"(),(n),(),(),(n,m),(n,m),(n,m)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def sigma_weighted_cross_general(weight,g,noise, delta, pp,pm,mm, out):
#     w = weight[0]
#     d2 = delta[0]**2
#     b2 = noise[0]**2
#     for i in range(len(g)):
#         g2 = g[i]**2
#         out[i] = (0.5 * (w**2 * (g2 + 1 + b2 + 2 * d2) - 4 * w * g[i]  + g2 + 1 - d2))
        
#         #correction terms, skipping p = 0 because it was computed above.
#         for p in range(1,pp.shape[1]):
#             tmp = (pp[i, p] + mm[i, p])*(_g(g, p)**2 + _g(g, p + i) * _g(g, p-i))
#             tmp += pm[i, p] *(_g(g, p + i)**2 + _g(g, p - i)**2 + _g(g, p)*_g(g, p + 2 * i) + _g(g, p)*_g(g, p - 2 * i))
            
#             tmp -= 2*w * (pp[i,p] + mm[i,p])* (_g(g,p+i)*_g(g,p) + _g(g,p-i)*_g(g,p)) 
#             tmp -= 2*w * pm[i,p]* (_g(g,p+i)*_g(g,p) + _g(g,p-i)*_g(g,p) + _g(g,p+i)*_g(g,p+2*i) + _g(g,p-i)*_g(g,p-2*i) ) 
        
#             tmp += w**2 * (pp[i,p] + mm[i,p])* (_g(g,p)**2 + _g(g,p+i)**2 + _g(g,p-i)**2)
#             tmp += w**2 * pm[i,p] * (_g(g,p)**2 +_g(g,p-i)**2 + _g(g,p+i)**2 + 0.5* _g(g,p+2*i)**2+ 0.5* _g(g,p-2*i)**2 )
            
#             out[i] = out[i] + 0.5 * tmp / (pp[i,0] + mm[i,0])
        
#         out[i] = out[i] ** 0.5
    
# @nb.guvectorize([(F[:],F[:],F[:],I64[:,:],F[:])],"(n),(n),(n),(n,m)->(n)",target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
# def sigma_weighted_auto_general(weight,g,noise, pp, out):
#     for i in range(len(g)):
#         w = weight[i]
#         b2 = noise[i]**2
#         g2 = g[i]**2
#         out[i] = 0.
#         out[i] = (0.5 * (w**2 * (g2 + 1 + b2) - 4 * w * g[i]  + g2 + 1))
        
#         #correction terms, skipping p = 0 because it was computed above.
#         for p in range(1,pp.shape[1]):
#             tmp = pp[i, p] * (_g(g, p)**2 + _g(g, p + i) * _g(g, p-i))

#             tmp -= 2*w * pp[i,p] * (_g(g,p+i)*_g(g,p) + _g(g,p-i)*_g(g,p)) 

#             tmp += w**2 * pp[i,p] * (_g(g,p)**2 + 0.5*_g(g,p+i)**2 + 0.5*_g(g,p-i)**2)

#             out[i] = out[i] + 0.5 * tmp / pp[i,0]
        
#         out[i] = out[i] ** 0.5    

