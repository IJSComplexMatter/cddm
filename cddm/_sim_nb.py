"""
Numba functions for sim module
"""

from __future__ import absolute_import, print_function, division

import numpy as np
import numba as nb
import math

from cddm.conf import U16,U8,F,I, NUMBA_TARGET, NUMBA_PARALLEL , NUMBA_CACHE, NUMBA_FASTMATH


#this is so simple that it is faster to compute in cpu mode
@nb.vectorize([F(F,F,F)], target = "cpu", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def mirror(x,x0,x1):
    """transforms coordinate x by flooring in the interval of [x0,x1]
    It performs x0 + (x-x0)%(x1-x0)"""
    #return x0 + (x-x0)%(x1-x0)
    
    # Most of the time there will be no need to do anything, 
    # so the implementation below is faster than floor implementation above
    
    if x0 == x1:
        return x
    while True: 
        if x < x0:
            x = x1 + x - x0
        elif x >= x1:
            x = x0 + x - x1
        else:
            break
    return x

@nb.njit()
def numba_seed(value):
    """Seed for numba random generator"""
    if NUMBA_TARGET != "cpu":
        print("WARNING: Numba seed not effective. Seeding only works with NUMBA_TARGET = 'cpu'")
    np.random.seed(value)
                          
@nb.vectorize([F(F,F,F)], target = "cpu", cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)        
def make_step(x,scale, velocity):
    """Performs random particle step from a given initial position x."""
    return x + np.random.randn()*scale + velocity   
     
@nb.jit([U8(I,F,I,F,F,U8),U16(I,F,I,F,F,U16)],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def psf_gauss(x,x0,y,y0,sigma,intensity):
    """Gaussian point-spread function. This is used to calculate pixel value
    for a given pixel coordinate x,y and particle position x0,y0."""
    return intensity*math.exp(-0.5*((x-x0)**2+(y-y0)**2)/(sigma**2))

@nb.jit([U8[:,:](U8[:,:],F[:,:],U8[:]),U16[:,:](U16[:,:],F[:,:],U16[:])], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                
def draw_points(im, points, intensity):
    """Draws pixels to image from a given points array"""
    data = points
    nparticles = len(data)
    assert points.shape[1] == 2
    assert nparticles == len(intensity)
    for j in range(nparticles):
        im[int(data[j,0]),int(data[j,1])] = im[int(data[j,0]),int(data[j,1])] + intensity[j] 
    return im  

@nb.jit([F(F,F,F)],nopython = True,cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def _adc_scale(frame, saturation, black_level):
    out = (frame - black_level)/saturation
    if out < 0:
        return 0.
    else:
        return out   
    
@nb.vectorize([F(F,F)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def shot_noise_gaussian(photon_noise, readout_noise):
    return photon_noise + readout_noise +  np.random.randn() * (photon_noise + readout_noise)**0.5

@nb.vectorize([F(F,F)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def shot_noise_poisson(photon_noise, readout_noise):
    out = 0.
    if photon_noise > 0.:
        out += np.random.poisson(photon_noise)
    if readout_noise > 0.:
        out += np.random.poisson(readout_noise)
    return out

@nb.vectorize([U16(F,F,F, U16), U8(F,F,F, U8)],  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def adc_clean(frame, saturation, black_level, maxv):
    out = (frame - black_level)/saturation * maxv
    if out < 0:
        return 0
    if out > maxv :
        return maxv
    else:
        return out   

@nb.vectorize([U8(F,F,F)],  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def adc_8bit(frame, scale, black_level):
    """Converts raw signal to unsigned int using 8bit conversion""" 
    out = _adc_scale(frame, scale, black_level)
    return int(out*256) if out < 1. else 255

@nb.vectorize([U16(F,F,F)],  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def adc_10bit(frame, scale, black_level):
    """Converts raw signal to unsigned int using 10bit conversion""" 
    out = _adc_scale(frame, scale, black_level)
    return int(out*1024) if out < 1. else 1023

@nb.vectorize([U16(F,F,F)],  cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH) 
def adc_12bit(frame, scale, black_level):
    """Converts raw signal to unsigned int using 12bit conversion""" 
    out = _adc_scale(frame, scale, black_level)
    return np.uint16(out*4096) if out < 1. else np.uint16(4095)

@nb.vectorize([U16(F,F,F)], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def adc_14bit(frame, scale, black_level):
    """Converts raw signal to unsigned int using 14bit conversion""" 
    out = _adc_scale(frame, scale, black_level)
    return int(out*16384) if out < 1. else 16383

@nb.vectorize([U16(F,F,F)], cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)    
def adc_16bit(frame, scale, black_level):
    """Converts raw signal to unsigned int using 16bit conversion""" 
    out = _adc_scale(frame, scale, black_level)
    return int(out*65536) if out < 1. else 65535

@nb.jit([U8[:,:](U8[:,:],F[:,:],U8[:],F[:]),U16[:,:](U16[:,:],F[:,:],U16[:],F[:])], nopython = True, parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                
def draw_psf(im, points, intensity, sigma):
    """Draws psf to image from a given points array"""
    assert points.shape[1] == 2
    nparticles = len(points)
    assert nparticles == len(intensity)
    assert nparticles == len(sigma)
    height, width = im.shape
    
    size = int(round(3*sigma.max()))
    for k in nb.prange(nparticles):
        h0,w0  = points[k,0], points[k,1]
        h,w = int(h0), int(w0) 
        for i0 in range(h-size,h+size+1):
            for j0 in range(w -size, w+size+1):
                p = psf_gauss(i0,h0,j0,w0,sigma[k],intensity[k])
                #j = j0 % width
                #i = i0 % height
                
                # slightly faster implementation of flooring
                if j0 >= width:
                    j= j0 - width
                elif j0 < 0:
                    j = width + j0
                else:
                    j = j0
                if i0>= height:
                    i = i0 - height
                elif i0 < 0:
                    i = height + i0
                else:
                    i = i0
                im[i,j] = im[i,j] + p
    return im  