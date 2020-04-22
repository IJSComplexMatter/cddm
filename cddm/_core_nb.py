"""
Low level numba functions
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import numba as nb
from cddm.conf import C,F, I64, NUMBA_TARGET, NUMBA_FASTMATH, NUMBA_CACHE
from cddm.fft import _fft, _ifft

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
    """Chooses data, randomly"""
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
    out[0] = a[0] #at the end we do not use this boundary data so it does not matter what we set here.
    out[n-1] = a[n-1]    
    for i in range(1,n-1):
        out[i] = 0.25*(a[i-1]+2*a[i]+a[i+1])

@nb.guvectorize([(F[:],F[:],F[:],F[:])],"(n),(),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def median_decrease(array, minv,maxv, out):
    """Performs median decrease filter. Each next element must be smaller or equal"""
    for i in range(len(array)):
        if i == 0:
            out[0] = maxv[0]
        elif array[i] < out[i-1]:
            out[i] = max(array[i],minv[0])
        else:
            out[i] = out[i-1]

@nb.guvectorize([(F[:],F[:],F[:],F[:])],"(n),(),()->(n)", target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def median_increase(array,minv, maxv,out):
    """Performs median increase filter. Each next element must be greater or equal"""
    for i in range(len(array)):
        if i == 0:
            out[0] = minv[0]
        elif array[i] > out[i-1]:
            out[i] = min(array[i],maxv[0])
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

@nb.vectorize([F(F,I64,C,C,F)],target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _normalize_ccorr_1(data, count, bg1, bg2, sq):
    tmp = data - 0.5 * sq 
    
    tmp = tmp/count
    
    d = (bg1.real - bg2.real)
    d2 = d*d
    d = (bg1.imag - bg2.imag)
    d2 = d2 + d*d
    
    return tmp + (0.5 * d2)

