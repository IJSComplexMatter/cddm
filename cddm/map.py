"""
Data mapping and k-averaging functions.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from cddm.decorators import deprecated


#from cddm.conf import F64,I64,U16, U16DTYPE, F, I, FDTYPE, I64DTYPE,F64DTYPE, NUMBA_CACHE

#@numba.jit([I64[:,:](I64,I64,F64)],nopython = True, cache = NUMBA_CACHE)
#def line_indexmap(height, width, angle):
#    shape = (height,width)
#    s = np.empty(shape, I64DTYPE)
#    s[...] = -1
#    y,x = (s.shape[0]//2+1),s.shape[1]
#
#    angle = angle * np.pi/180
#    absangle = abs(angle)
#    
#    if absangle <= math.atan2(y,x) and absangle >=0:
#        for j in range(x):
#            i = - int(round(math.tan(angle)*j))
#            s[i,j] = j 
#    elif absangle > math.atan2(y,x) and absangle <= 90:
#        if angle > 0:
#            angle = np.pi/2 - angle 
#            for r,i in enumerate(range(y)):
#                j = int(round(math.tan(angle)*i))
#                s[-i,j] = r
#        else:
#            angle = angle+ np.pi/2
#            for r,i in enumerate(range(y)):
#                j =  int(round(math.tan(angle)*i))
#                s[i,j] = r            
#    return s         

@deprecated("Use fft2_sector_indexmap instead.")
def sector_indexmap(kmap, anglemap, angle = 0., sector = 30., kstep = 1.):
    """Builds indexmap array of integers ranging from -1 (invalid data)
    and positive integers. Each non-negative integer is a valid k-index computed
    from sector parameters.
    
    Parameters
    ----------
    kmap : ndarray
        Size of wavevector at each (i,j) indices in the rfft2 ouptut data.
    anglemap : ndarray
        Angle of the wavevector at each (i,j) indices in the rfft2 ouptut data.      
    angle : float
        Mean angle of the sector in degrees (-90 to 90).
    sector : float
        Width of the sector in degrees (between 0 and 180) 
    kstep : float, optional
        k resolution in units of minimum step size.
        
    Returns
    -------
    map : ndarray
        Ouput array of non-zero valued k-indices where data is valid, -1 elsewhere.
    """
    if angle < -90 or angle > 90:
        raise ValueError("Wrong angle value")
    if sector <= 0 or sector > 180:
        raise ValueError("Wrong sector value")
    kmax = kmap.max()
    ks = np.arange(0, kmax + kstep, kstep)
    
    out = np.empty(kmap.shape, int)
    out[...] =-1
    
    high = np.pi/180*(angle + sector/2)
    low  = np.pi/180*(angle - sector/2)
    
    m = np.pi/2
    negate = False
        
    if high > m:
        low, high = high%(-m), low
        negate = True
    elif low < -m:
        high, low = low%m, high
        negate = True
        
    if negate:
        fmask = (anglemap <= high) & (anglemap > low)
        fmask = np.logical_not(fmask)
    else:
        fmask = (anglemap < high) & (anglemap >= low)
    
    for i,k in enumerate(ks):
        kmask = (kmap <= (k + kstep/2.)) &  (kmap > (k - kstep/2.))
        mask = kmask & fmask
        out[mask] = i
    half = kmap.shape[0]//2+1
    #no new information here... so remove it
    out[half:,0] = -1
    #k = (0,0) is present regardless of angle, sector
    out[0,0] = 0
    return out

def _fft2_sector_indexmap(kmap, anglemap, ks, angle = 0., sector = 30., kstep = 1., include_origin = True):
    if angle < -180 or angle > 180:
        raise ValueError("Wrong angle value")
    if sector <= 0 or sector > 180:
        raise ValueError("Wrong sector value")
    
    out = np.empty(kmap.shape, int)
    out[...] =-1
    
    high = np.pi/180*(angle + sector/2)
    low  = np.pi/180*(angle - sector/2)
    
    m = np.pi
    negate = False
        
    if high > m:
        low, high = high%(-m), low
        negate = True
    elif low < -m:
        high, low = low%m, high
        negate = True
        
    if negate:
        fmask = (anglemap <= high) & (anglemap > low)
        fmask = np.logical_not(fmask)
    else:
        fmask = (anglemap < high) & (anglemap >= low)
    
    for i,k in enumerate(ks):
        kmask = (kmap <= (k + kstep/2.)) &  (kmap > (k - kstep/2.))
        mask = kmask & fmask
        out[mask] = i

    #k = (0,0) is present regardless of angle, sector
    if include_origin == True:
        out[0,0] = 0
    return out


def fft2_sector_indexmap(kmap, anglemap, angle = 0., sector = 30., kstep = 1.):
    """Builds indexmap array of integers ranging from -1 (invalid data)
    and positive integers. Each non-negative integer is a valid k-index computed
    from sector parameters.
    
    Parameters
    ----------
    kmap : ndarray
        Size of wavevector at each (i,j) indices in the rfft2 ouptut data.
    anglemap : ndarray
        Angle of the wavevector at each (i,j) indices in the rfft2 ouptut data.      
    angle : float
        Mean angle of the sector in degrees (-180 to 180).
    sector : float
        Width of the sector in degrees (between 0 and 180) 
    kstep : float, optional
        k resolution in units of minimum step size.
        
    Returns
    -------
    map : ndarray
        Ouput array of non-zero valued k-indices where data is valid, -1 elsewhere.
    """
    kmax = kmap.max()
    ks = np.arange(0, kmax + kstep, kstep)
    return _fft2_sector_indexmap(kmap, anglemap, ks, angle = angle, sector = sector, kstep = kstep)

def fft2_sector_mask(kmap, anglemap, k, angle = 0., sector = 30., kwidth = 1.):
    ks = (k,)
    out = _fft2_sector_indexmap(kmap, anglemap, ks, angle = angle, sector = sector, kstep = kwidth, include_origin = False)
    # Only one k-value, make that value +1, empty data is -1, which makes it zero
    out+=1
    return np.asarray(out,bool)

def _adjust_index(i,n):
    n_half = n/2 
    if i > n_half:
        i = i%(-n_half)
    if i < -n_half:
        i = i%(n_half)  
    return i
    
def circle_mask(ii,jj, i, j, radius = 1, inclusive = False):
    height, width = ii.shape
    i = _adjust_index(i,height)
    j = _adjust_index(j,width)
    mi = ii - i
    mj = jj - j
    if inclusive == True:
        m = np.sqrt(mi**2 + mj**2) <= radius
    else:
        m = np.sqrt(mi**2 + mj**2) < radius
    return m
    
def _get_steps_size_half(kisize,kjsize,shape):
    height, width = (1, 1) if shape is None else shape
    
    if kjsize is None:
        if shape is None:
            raise ValueError("Shape is a required argument")
        kjsize = width//2+1

    if kisize is None:
        if shape is None:
            raise ValueError("Shape is a required argument")
        kisize = height
    
    kistep = 1.
    kjstep = 1.
    
    if height > width:
        kjstep = height/width
    else:
        kistep = width/height
    return (kisize, kistep), (kjsize, kjstep)

def _get_steps_size_full(kisize,kjsize,shape):
    if kjsize is None:
        if shape is None:
            raise ValueError("Shape is a required argument")
        kjsize = shape[1]
    return _get_steps_size_half(kisize,kjsize,shape)

def ij2rf(ki,kj):
    y,x = ki,kj
    x2, y2  = x**2, y**2
    return (x2 + y2)**0.5 , np.arctan2(-y,x)      

def rf2ij(k,f):
    return -k*np.sin(f), k*np.cos(f)

def rfft2_kangle(kisize = None, kjsize = None, shape = None):
    """Build k,angle arrays based on the size of the rfft2 data. 
    
    Parameters
    ----------
    kisize : int
        i-size of the cropped rfft2 data. 
    kjsize : int
        j-size of the cropped rfft2 data
    shape : (int,int)
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
        
    Returns
    -------
    k, angle :  ndarray, ndarray
        k, angle arrays 
    """
    ki,kj = rfft2_grid(kisize = kisize, kjsize = kjsize, shape = shape)
    return ij2rf(ki,kj)
            
def fft2_kangle(kisize = None, kjsize = None, shape = None):
    """Build k,angle arrays based on the size of the fft2 data. 
    
    Parameters
    ----------
    kisize : int
        i-size of the cropped fft2 data. 
    kjsize : int
        j-size of the cropped fft2 data
    shape : (int,int)
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
        
    Returns
    -------
    k, angle, ndarray, ndarray
        k, angle arrays 
    """
    ki,kj = fft2_grid(kisize = kisize, kjsize = kjsize, shape = shape)
    return ij2rf(ki,kj)

def _grid_half(kisize_step, kjsize_step):
    (kisize, kistep), (kjsize, kjstep) = kisize_step, kjsize_step
    ki,kj = np.meshgrid(np.fft.fftfreq(kisize)*kisize*kistep,np.arange(kjsize)*kjstep, indexing = "ij")  
    return ki,kj

def _grid_full(kisize_step, kjsize_step):
    (kisize, kistep), (kjsize, kjstep) = kisize_step, kjsize_step
    ki,kj = np.meshgrid(np.fft.fftfreq(kisize)*kisize*kistep,np.fft.fftfreq(kjsize)*kjsize*kjstep, indexing = "ij")  
    return ki,kj

def rfft2_grid(kisize = None, kjsize = None, shape = None):
    """Build ki,kj coordinate arrays based on the size of the rfft2. 
    
    Parameters
    ----------
    kisize : int
        i-size of the cropped rfft2 data. 
    kjsize : int
        j-size of the cropped rfft2 data
    shape : (int,int)
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
        
    Returns
    -------
    ki, kj, ndarray, ndarray
        ki,kj coordinate arrays 
     
    """
    kisize_step, kjsize_step = _get_steps_size_half(kisize,kjsize,shape)
    return _grid_half(kisize_step, kjsize_step )

def fft2_grid(kisize = None, kjsize = None, shape = None):
    """Build ki,kj coordinate arrays based on the size of the fft2. 
    
    Parameters
    ----------
    kisize : int
        i-size of the cropped fft2 data. 
    kjsize : int
        j-size of the cropped fft2 data
    shape : (int,int)
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
        
    Returns
    -------
    ki, kj, ndarray, ndarray
        ki,kj coordinate arrays 
     
    """
    kisize_step, kjsize_step = _get_steps_size_full(kisize,kjsize,shape)
    return _grid_full(kisize_step, kjsize_step )

def grid(shape):
    kisize, kjsize = shape
    return fft2_grid(kisize, kjsize, shape)

def _k_select_complex(data, k, indexmap, kmap, computed_mask, masked_data):
    mask = (indexmap == int(round(k)))
    if computed_mask is not None:
        mask = mask & computed_mask
    ks = kmap[mask]
    if masked_data == True:
        mask = mask[computed_mask]
    if len(ks) > 0:
        #at least one k value exists
        data_avg = np.nanmean(data[...,mask,:], axis = -2)
        k_avg = ks.mean()
        return k_avg, data_avg  
    else:
        #no k values
        return None
    
def _k_select_real(data, k, indexmap, kmap, computed_mask, masked_data):
    mask = (indexmap == int(round(k)))
    ks = kmap[mask]
    mask = as_rfft2_mask(mask)
    conj_mask = as_rfft2_conj(mask)
    
    if computed_mask is not None:
        mask = mask & computed_mask
        conj_mask = conj_mask & computed_mask
        
    
    if masked_data == True:
        mask = mask[computed_mask]
        conj_mask = conj_mask[computed_mask]
    if len(ks) > 0:
        #at least one k value exists
        conj_mask = conj_mask[mask]
        masked_data = data[...,mask,:]
        masked_data[...,conj_mask,:] = np.conj(masked_data[...,conj_mask,:])
        data_avg = np.nanmean(masked_data, axis = -2)
        k_avg = ks.mean()
        return k_avg, data_avg  
    else:
        #no k values
        return None
    
def k_point_kernel(window, ki,kj):
    height, width = window.shape
    k_window = np.zeros((height,width))
    k_window[ki,kj] = 1
    #sine wave
    real_window = np.fft.fft2(k_window)
    real_window *= window
    return real_window    

@deprecated("Use fft2_k_indexmap instead.")    
def k_indexmap(kisize,kjsize, angle = 0, sector = 5, kstep = 1., shape = None):
    """Builds indexmap array of integers ranging from -1 (invalid data)
    and positive integers. Each non-negative integer is a valid k-index computed
    from sector parameters.
    
    Parameters
    ----------
    kisize : int
        Height of the fft data
    kjsize : int
        Width of the fft data
    angle : float
        Mean angle of the sector in degrees (-90 to 90).
    sector : float
        Width of the sector in degrees (between 0 and 180) 
    kstep : float, optional
        k resolution in units of minimum step size.
    shape : (int,int), optional
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
        
    Returns
    -------
    map : ndarray
        Ouput array of non-zero valued k-indices where data is valid, -1 elsewhere.
    """
    kmap, anglemap = rfft2_kangle(kisize, kjsize,shape)
    indexmap = sector_indexmap(kmap, anglemap, angle, sector, kstep) 
    return indexmap

def as_fft2_shape(kisize = None, kjsize = None, shape = None):
    if kjsize == None:
        if shape is None:
            raise ValueError("shape is a required argument")
        _, kjsize = shape
    else:
        kjsize = kjsize * 2 - 1
        if shape is not None:
            _, width = shape 
            if kjsize == width + 1:
                #ok, thist must be original data shape - no cropping, so set to full width.
                kjsize = width
            elif kjsize > width:
                raise ValueError("kjsize incompatible with input data shape")
        
    if kisize is None:
        if shape is None:
            raise ValueError("shape is a required argument")
        kisize, _ = shape  
    else:
        if shape is not None:
            height, _ = shape 
            if kisize > height:
                raise ValueError("kisize incompatible with input data shape")        
    return kisize, kjsize

def as_rfft2_mask(a, axes = (-2,-1)):
    """Converts fft2 (boolean) mask array into rfft2 (boolean) mask array.
    """
    m1 = as_rfft2_data(a, axes)
    m2 = as_rfft2_conj(a, axes)
    return np.logical_or(m1,m2)

def as_rfft2_conj(a, axes = (-2,-1)):
    """Converts data from `fft2 space` to `rfft2 conjugate half space`.
    
    It stores the missing part in the rfft2 data from the full fft2 data.
    The shape of the ouptut is the same as that of the as_rfft2_data. Note that
    you should only access non-zero frequencies. First column does not hold any data.
    
    
    Examples
    --------
    >>> a = np.random.randn(32,32)
    >>> f = np.fft.fft2(a)
    >>> rf = np.fft.rfft2(a)
    >>> cf = as_rfft2_conj(f)
    >>> np.allclose(cf[2,5], f[-2,-5])
    True
    """
    ax1,ax2 = tuple(sorted(axes))
    
    slices_in = [slice(None)]*a.ndim
    slices_out = [slice(None)]*a.ndim
    
    w = a.shape[ax2]
    #width of the half-space data
    whalf = w//2 + 1
    
    slices_in[ax2] = slice(0,whalf) 
    
    m2 = np.empty_like(a[tuple(slices_in)])
    
    #the negative values
    if m2.dtype == bool:
        #set False to indicate invalid data
        m2[...] = False
    else:
        try:
            #set np.nan to indicate invalid data
            m2[...] = np.nan
        except:
            #non-float and non-bolean type, so it must be int type, set -1 to indicate ivalid data
            m2[...] = -1
            
    # now set valid data       
    #m2[...,1:,1:] = a[...,-1:0:-1,-1:-whalf:-1]
    slices_in[ax1] = slice(-1,0,-1)
    slices_in[ax2] = slice(-1,-whalf,-1) 
    slices_out[ax1] = slice(1,None)
    slices_out[ax2] = slice(1,None)

    m2[tuple(slices_out)] = a[tuple(slices_in)]
    
    #m2[...,0,1:] = a[...,0,-1:-whalf:-1]
    slices_in[ax1] = 0
    slices_in[ax2] = slice(-1,-whalf,-1) 
    slices_out[ax1] = 0
    slices_out[ax2] = slice(1,None)
    
    m2[tuple(slices_out)] = a[tuple(slices_in)]

    return m2

def as_rfft2_data(a, axes = (-2,-1)):
    """Converts data from `fft2 space` to `rfft2 half space`.
    
    Examples
    --------
    >>> a = np.random.randn(32,32)
    >>> f = np.fft.fft2(a)
    >>> rf = np.fft.rfft2(a)
    >>> hf = as_rfft2_data(f)
    >>> np.allclose(hf,rf)
    True
    """
    ax1,ax2 = tuple(sorted(axes))
    
    slices_in = [slice(None)]*a.ndim
    
    w = a.shape[ax2]
    #width of the half-space data
    whalf = w//2 + 1
    
    slices_in[ax2] = slice(0,whalf)
    
    m1 = a[tuple(slices_in)]
    return m1

def from_rfft2_data(a, shape = None, axes = (-2,-1)):
    """Converts halfspace rfft2 data to fullspace fft2 data. 

    Examples
    --------
    >>> a = np.random.randn(32,32)
    >>> f = np.fft.fft2(a)
    >>> rf = np.fft.rfft2(a)
    >>> hf = as_rfft2_data(f)
    >>> np.allclose(hf,rf)
    True
    >>> np.allclose(from_rfft2_data(hf, shape = (32,32)), f)
    True
    """
    ax1,ax2 = tuple(sorted(axes))
    
    slices_in = [slice(None)]*a.ndim
    slices_out = [slice(None)]*a.ndim
    
    kisize, kjsize = a.shape[ax1], a.shape[ax2]
    
    newh, neww = as_fft2_shape(kisize = kisize, kjsize = kjsize, shape = shape)
    new_shape = list(a.shape)
    new_shape[ax1] = newh
    new_shape[ax2] = neww
    
    
    out = np.empty(shape = new_shape, dtype = a.dtype)
    whalf = neww//2 + 1
    
    #out[...,:,0:whalf] = a
    slices_out[ax2] = slice(0,whalf)
    out[tuple(slices_out)] = a
    
    #in case shape[1] is odd this is same as shape[1]//2 + 1, else it is lower by one
    whalf2 = (neww + 1)//2
    
    
    #out[...,1:,whalf2:] = np.conj(a[...,-1:0:-1,-1:-whalf:-1])
    slices_in[ax1] = slice(-1,0,-1)
    slices_in[ax2] = slice(-1,-whalf,-1)
    slices_out[ax1] = slice(1,None)
    slices_out[ax2] = slice(whalf2,None)
    
    if np.iscomplexobj(a):
        out[tuple(slices_out)] = np.conj(a[tuple(slices_in)])
    else:
        out[tuple(slices_out)] = a[tuple(slices_in)]
    
    #out[...,0,whalf2:] = np.conj(a[...,0,-1:-whalf:-1])
    slices_in[ax1] = 0
    slices_in[ax2] = slice(-1,-whalf,-1)
    slices_out[ax1] = 0
    slices_out[ax2] = slice(whalf2,None)
    
    if np.iscomplexobj(a):
        out[tuple(slices_out)] = np.conj(a[tuple(slices_in)])
    else:
        out[tuple(slices_out)] = a[tuple(slices_in)]
        
    return out
    

def fft2_k_indexmap(kisize,kjsize, angle = 0, sector = 5, kstep = 1., shape = None, mode = "rfft2"):
    """Builds indexmap array of integers ranging from -1 (invalid data)
    and positive integers. Each non-negative integer is a valid k-index computed
    from the sector parameters.
    
    Parameters
    ----------
    kisize : int
        Height of the fft2 or rfft2 data
    kjsize : int
        Width of the fft2 or rfft2 data
    angle : float
        Mean angle of the sector in degrees (-90 to 90).
    sector : float
        Width of the sector in degrees (between 0 and 180) 
    kstep : float, optional
        k resolution in units of minimum step size.
    shape : (int,int), optional
        Shape of the original data. This is used to calculate step size. If not
        given, rectangular data is assumed (equal steps).
    mode : str, optional
        Describes inpud data mode. Either 'rfft2' or 'fft2'.
        
    Returns
    -------
    map : ndarray
        A fullsize ouput array of non-zero valued k-indices where data is valid, -1 elsewhere.
    """
    if mode == "rfft2":
        kisize, kjsize = as_fft2_shape(kisize, kjsize, shape)
        
    kmap, anglemap = fft2_kangle(kisize, kjsize, shape)
    indexmap = fft2_sector_indexmap(kmap, anglemap, angle, sector, kstep) 
    return indexmap

def plot_indexmap(graph, mode = "rfft2"):
    """Plots indexmap array. If mode == "rfft2" it plots half-space data,
    If mode =="fft2" it performs plotting of full-spaced data."""
    import matplotlib.pyplot as plt
    if mode == "rfft2":
        extent=[0,graph.shape[1],graph.shape[0]//2+1,-graph.shape[0]//2-1]
        plt.imshow(np.fft.fftshift(graph,0), extent = extent)
    elif mode == "fft2":
        extent=[-graph.shape[1]//2-1,graph.shape[1]//2+1,graph.shape[0]//2+1,-graph.shape[0]//2-1]
        plt.imshow(np.fft.fftshift(graph), extent = extent)        
    else:
        raise ValueError("Invalid mode")
        
def k_select(data, angle , sector = 5, kstep = 1, k = None, shape = None, mask = None, mode = "rfft2"):
    """k-selection and k-averaging of normalized (and merged) correlation data.
    
    This function takes (...,i,j,n) correlation data and performs k-based selection
    and averaging of the data. If you analyzed masked video, you must provide
    the mask.
    
    Parameters
    ----------
    deta : array_like
        Input correlation data of shape (...,i,j,n)
    angle : float
        Angle between the j axis and the direction of k-vector
    sector : float, optional
        Defines sector angle, k-data is avergaed between angle+sector and angle-sector angles
    kstep : float, optional
        Defines an approximate k step in pixel units
    k : float or list of floats, optional
        If provided, only return at a given k value (and not at all non-zero k values)
    shape : tuple
        Shape of the original video frame. If shape is not rectangular, it must
        be provided.
    mask : ndarray, optional
        A boolean array. This is the mask used in :func:`.video.mask`.
        
    Returns
    -------
    out : iterator, or tuple
        If k s not defined, this is an iterator that yields a tuple of (k_avg, data_avg)
        of actual (mean) k and averaged data. If k is a list of indices, it returns
        an iterator that yields a tuple of (k_avg, data_avg) for every non-zero data
        if k is an integer, it returns a tuple of (k_avg, data_avg) for a given k-index.
    """
    
    assert sector > 0
    
    #whether data was masked or not
    masked_data = False
    
    if mask is not None:
        kisize, kjsize = mask.shape
        #if shapes of mask and frame do not match
        if data.shape[-3:-1] != (kisize, kjsize):
            masked_data = True
        #else assume, data was not masked
    else:
        kisize, kjsize = data.shape[-3:-1]
        
    if mode == "rfft2":
        kisize, kjsize = as_fft2_shape(kisize, kjsize, shape)
        _k_select = _k_select_real
    else:
        _k_select = _k_select_complex
    
    kmap, anglemap = fft2_kangle(kisize, kjsize, shape)
    indexmap = fft2_sector_indexmap(kmap, anglemap, angle, sector, kstep)

    if k is None:
        kdim = indexmap.max() + 1
        k = np.arange(kdim)
    try:
        iterator = (_k_select(data, kk, indexmap, kmap, mask, masked_data) for kk in k)
        return tuple((data for data in iterator if data is not None))       
    except TypeError:
        #not iterable
        return _k_select(data, k, indexmap, kmap, mask)

class Mask:
    def __init__(self, mask_shape, frame_shape = None,  data_space = "rfft2"):
        self.mask_shape = mask_shape
        self.frame_shape = frame_shape
        self.data_space = data_space
        
        kisize, kjsize = self.mask_shape
        if self.data_space == "rfft2":
            kisize, kjsize = as_fft2_shape(kisize, kjsize,shape = self.frame_shape)

        self.imap, self.jmap = fft2_grid(kisize, kjsize, shape = self.frame_shape)        
        self.kmap, self.anglemap = fft2_kangle(kisize = kisize, kjsize = kjsize, shape = self.frame_shape)
                     
    def set_sector(self,center, sector = 5, kwidth = 1, coordinates = "ij"):
        if self.coordinates == "ij":
            k,angle = ij2rf(*center)
            angle = angle * 180/np.pi
        else:
            k, angle = center
        mask = fft2_sector_mask(self.kmap, self.anglemap, k = k, angle = angle,sector = sector, kwidth = kwidth)
        return self.set_mask(mask)
        
    def set_circle(self, center, radius = 1, coordinates = "rf"):
        if coordinates == "ij":
            i,j = center
        else:
            k, angle = center 
            i, j = rf2ij(k,angle*np.pi/180.)
        mask = circle_mask(self.imap, self.jmap, i = i,j = j, radius = radius)
        return self.set_mask(mask)
        
    def set_mask(self, mask):        
        if mask is not None:
            self._mask = mask
            if self.data_space == "rfft2":
                self._data_mask = as_rfft2_mask(mask)
                self._conj_mask = as_rfft2_conj(mask)
            else:
                self._conj_mask = None
                self._data_mask = mask           
        else:
            self._mask = None
            self._conj_mask = None  
            self._data_mask = None
            
    @property        
    def mask(self):
        return self._mask
        
    @property        
    def conj_mask(self):
        if self.data_space == "rfft2":
            return self._conj_mask  
        
    @property        
    def data_mask(self):
        return self._data_mask        

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    k,f = fft2_kangle(129,129, (256,256))
    m = fft2_sector_indexmap(k,f, angle = 0,sector = 30)
    plt.imshow(np.fft.fftshift(m))
    
    kmask = m == 32
    plt.figure()
    plt.imshow(np.fft.fftshift(kmask))
    
    m = fft2_sector_mask(k,f, k = 0.4, angle = 0,sector = 30)
    plt.figure()
    plt.imshow(m)
    
    ii, jj = grid((256,256))
    
    m = circle_mask(ii,jj,-5,-4,1)
    plt.figure()
    plt.imshow(m)   
    
    

    
    