#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:11:06 2019

@author: andrej
"""

import numba
import math
import numpy as np

from cddm.conf import F64,I64,U16, U16DTYPE, F, I, FDTYPE, I64DTYPE,F64DTYPE

@numba.jit([I64[:,:](I64,I64,F64)],nopython = True, cache = True)
def line_indexmap(height, width, angle):
    shape = (height,width)
    s = np.empty(shape, I64DTYPE)
    s[...] = -1
    y,x = (s.shape[0]//2+1),s.shape[1]

    phi = angle * np.pi/180
    absangle = abs(phi)
    
    if absangle <= math.atan2(y,x) and absangle >=0:
        for j in range(x):
            i = - int(round(math.tan(phi)*j))
            s[i,j] = j 
    elif absangle > math.atan2(y,x) and absangle <= 90:
        if phi > 0:
            phi = np.pi/2 - phi 
            for r,i in enumerate(range(y)):
                j = int(round(math.tan(phi)*i))
                s[-i,j] = r
        else:
            phi = phi+ np.pi/2
            for r,i in enumerate(range(y)):
                j =  int(round(math.tan(phi)*i))
                s[i,j] = r            
    return s         


@numba.jit([I64[:,:](I64,I64,F64,F64, F64)],nopython = True,cache = True)
def sector_indexmap(height, width, angle, sector, kstep):
    shape = (height,width)
    s = np.empty(shape, I64DTYPE)
    s[...] = -1
    y,x = (s.shape[0]//2+1),s.shape[1]
    maxr = max(x,y)-1
    absangle = abs(angle)
    
    if absangle <= math.atan2(y,x) and absangle > 0:
        y = int(round(math.tan(angle/180.*np.pi)*x))
    elif absangle > math.atan2(y,x) and absangle <= 90:
        x = int(round(math.tan((90-angle)/180.*np.pi)*y))
    dim = max(x,y)
    if kstep <= 0.:
        factor = 1.0*dim/(x**2 + y**2)**0.5
    else:
        factor = 1./kstep
    angle = np.pi/180*angle
    fmax1 = angle + sector/360.*np.pi
    fmin1 = angle - sector/360.*np.pi  
    if angle > 0:
        fmax2 = fmax1 - np.pi
        fmin2 = fmin1 - np.pi  
    else:
        fmax2 = fmax1 + np.pi
        fmin2 = fmin1 + np.pi       
    for j in range(-s.shape[0]//2,s.shape[0]//2):
        y2 = j**2
        for k in range(s.shape[1]):
            x2 = k**2   
            kdim =  math.sqrt((y2+x2))   
            r = int(round(kdim*factor)) #or should be between 0 and dim...
            angle = math.atan2(-j,k) #reverse first axis because arrays have topleft origin
            #print (j,k, angle, r)
            if (r > 0 and (angle >= fmin1 and angle <= fmax1)):
                if not (k == 0 and j > 0): # [i,0] = [-i,0]* so skip this
                    s[j,k] = r
            elif (r > 0 and (angle >= fmin2 and angle <= fmax2)):
                if not (k == 0 and j > 0): # [-i,0] = [i,0]* so skip this
                    s[j,k] = r
            elif r == 0:
                s[j,k] = 0
            
    return s  
 
@numba.guvectorize([(I64[:,:], I64[:], F64[:])], "(i,j),()->()")
def q_vec(indexmap, qindex, out):
    n = np.zeros((out.shape[0],), I64DTYPE)
    for j in range(-(indexmap.shape[0]//2), indexmap.shape[0]//2):
        for k in range(indexmap.shape[1]):
            r = indexmap[j,k] 
            if r == qindex[0]:
                x2 = k**2 
                y2 = j**2
                kdim =  math.sqrt((y2+x2)) 
                out[r] = out[r] + kdim
                n[r] = n[r] + 1
    for i in range(out.shape[0]):
        if n[i] !=0:
            out[i] = out[i] / n[i]
        else:
            out[i] = np.nan


#
#
#@numba.jit([F64[:](I64[:,:])],nopython = True,cache = True)    
#def _calc_kvec(indexmap):
#    dim = indexmap.max()+1
#    out = np.zeros((dim,), F64DTYPE)
#    n = np.zeros((dim,), I64DTYPE)
#    for j in range(-(indexmap.shape[0]//2), indexmap.shape[0]//2):
#        y2 = j**2
#        for k in range(indexmap.shape[1]):
#            x2 = k**2 
#            r = indexmap[j,k]
#            if r != 0:  
#                kdim =  math.sqrt((y2+x2)) 
#                out[r] = out[r] + kdim
#                n[r] = n[r] + 1
#    for i in range(dim):
#        if n[i] !=0:
#            out[i] = out[i] / n[i]
#        else:
#            out[i] = np.nan
#    return out
#    
#@numba.jit(["float32[:,:](float32[:,:,:],uint16[:,:])", "float64[:,:](float64[:,:,:],uint16[:,:])",
#            "complex64[:,:](complex64[:,:,:],uint16[:,:])", "complex128[:,:](complex128[:,:,:],uint16[:,:])"],nopython = True,cache = True)
#def _average(s,indexmap):
#    """Averages data based on the indexmap array.
#    """
#    dim = indexmap.max()+1
#    out = np.zeros((dim,s.shape[2]),s.dtype)
#    n = np.zeros((dim,),I64DTYPE)
#    for j in range(s.shape[0]):
#        for k in range(s.shape[1]):
#            r = indexmap[j,k]
#            if r != 0 or (j == 0 and k == 0):
#                for i in range(s.shape[2]):
#                    out[r,i] = out[r,i] + s[j,k,i]
#                n[r] = n[r]+1
#               
#    for i in range(out.shape[0]):
#        if n[i] != 0:
#            for j in range(out.shape[1]):
#                out[i,j] = out[i,j]/n[i]
#    return out
#
#            
#def q_average(s,angle,sector, kstep):
#    """Average input array over k-vectors at a given angle"""
#    indexmap = sector_indexmap(s.shape[0],s.shape[1],angle,sector,kstep)
#    out = _average(s,indexmap)
#    kvec = _calc_kvec(indexmap)
#    return kvec, out   