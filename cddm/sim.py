"""
C-DDM video simulator for two-dimensional Brownian motion of spherical particles.

You can use this script to simulate C-DDM experiment. You define number of particles
to simulate and the region of interest (simulation box). Particles are randomly
placed in the box, simulation is done using periodic boundary conditions - when 
the particle reaches the edge of the frame it is mirrored on the other side. 
Then optical microscope image capture is simulated by drawing a point spread function to the 
image at the calculated particle positions at a predefined time of interest for 
both cameras. 

The frame grabber is done using iterators to reduce memory requirements. You can
analyze video frame by frame with minimal memory requirement.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
from cddm.conf import FDTYPE

from cddm._sim_nb import mirror, numba_seed, make_step, draw_points, draw_psf, shot_noise_poisson, shot_noise_gaussian, adc_clean
import cddm._sim_nb as _sim_nb

def _inspect_frame_dtype(dtype, intensity):
    dtype = np.dtype(dtype)
    if dtype not in ("uint8", "uint16"):
        raise ValueError("Unsupported dtype")
    if intensity > 255 and dtype != "uint16" or intensity < 0:
        raise ValueError("Invalid intensity for dtype {}".format(dtype))
    return dtype

def form_factor(window, sigma = 3, intensity = 5, navg = 10, dtype = "uint8", mode = "rfft2"):
    """Computes point spread function form factor.
    
    Draws a PSF randomly in the frame and computes FFT, returns average absolute
    of the FFT.
    
    Parameters
    ----------
    window : ndarray
        A 2D window function used in the analysis. Set this to np.ones if you do
        not use one. 
    sigma : float
        Sigma of the PSF of the image
    intensity : unsigned int
        Intensity value
    navg : int
        Specifies mesh size for averaging.
    dtype : np.dtype
        One of "uint8" or "uint16". Defines output dtype of the image.
    mode : str, optional
        Either 'rfft2' (default) or 'fft2'. 
        
    Returns
    -------
    out : ndarray
        Average absolute value of the FFT of the PSF.
    """
    out = 0.
    window = np.asarray(window)
    shape = window.shape
    dtype = _inspect_frame_dtype(dtype, intensity)
    
    for i in range(navg):
        for j in range(navg):
            im = np.zeros(shape, dtype)
            coord = (np.array(((i,j),)) * np.array(shape))*1.0 / navg
            im = draw_psf(im, coord,np.array([intensity], dtype),np.array([sigma], FDTYPE))
            im = im*window
            if mode == 'rfft2':
                out += np.abs(np.fft.rfft2(im))**2
            elif mode == "fft2":
                out += np.abs(np.fft.fft2(im))**2
            else:
                raise ValueError("Unknown mode.")
             
    return (out/navg**2)**0.5

def seed(value):
    """Seed for numba and numpy random generator"""
    np.random.seed(value)
    numba_seed(value)

def brownian_walk(x0, count = 1024, shape = (256,256), delta = 1, dt = 1, velocity = (0.,0.)):
    """Returns an brownian walk iterator.
     
    Given the initial coordinates x0, it callculates and yields next `count` coordinates.
    
    Parameters
    ----------
    x0 : array-like
        A list of initial coordinates (i, j) of particles (in pixel units).
    count : int
        Number of simulation steps.
    shape : (int,int)
        Shape of the simulation region in pixels.
    delta : float
        Defines an average step in pixel coordinates (when dt = 1).
    dt : float, optional
        Simulation time step (1 by default).
    velocity : (float,float), optional
        Defines an average velocity (vi,vj) in pixel coordinates per unit time step
        (when dt = 1).
        
    Yields
    ------
    coordinates : ndarray
        Coordinates 2D array for the particles. The second axis is the x,y coordinate.
    """ 
    
    x = np.asarray(x0,FDTYPE)
    try:            
        particles, xy = x.shape
    except:
        raise ValueError("Wrong shape of input coordinats.")
    #step size
    scale=delta*np.sqrt(dt)
    velocity = np.asarray(velocity, FDTYPE)*dt
    scale = np.asarray(scale, FDTYPE)
    
    if len(shape) != 2:
        raise ValueError("Wrong `shape`, must be 2D.")
    
    x1, x2 = 0, np.asarray(shape,FDTYPE)
    
    x = mirror(x,x1,x2) #make sure we start in the box
    
    
    for i in range(count):
        yield x
        x = make_step(x,scale, velocity) 
        x = mirror(x,x1,x2)
        
def brownian_particles(count = 500, shape = (256,256),num_particles = 100, delta = 1, dt = 1,velocity = 0., x0 = None):
    """Coordinates generator of multiple brownian particles.
    
    Builds particles randomly distributed in the computation box and performs
    random walk of coordinates.
    
    Parameters
    ----------
    count : int
        Number of steps to calculate
    shape : (int,int)
        Shape of the box
    num_particles : int
        Number of particles in the box
    delta : float
        Step variance in pixel units (when dt = 1)
    dt : float
        Time resolution
    velocity : float
        Velocity in pixel units (when dt = 1) 
    x0 : array-like
        A list of initial coordinates
        
    Yields
    ------
    coordinates : ndarray
        Coordinates 2D array for the particles. The second axis is the x,y coordinate.
        Length of the array equals number of particles.
    """
    if x0 is None:
        x0 = np.asarray(np.random.rand(int(num_particles),2)*np.array(shape),FDTYPE)
    else:
        x0 = np.asarray(x0,FDTYPE)
        if len(x0) != num_particles:
            raise ValueError("Wrong length of initial coordinates x0")
 
    v0 = np.zeros_like(x0)
    v0[:,:] = velocity
    for data in brownian_walk(x0,count,shape,delta,dt,v0):
        yield data
             
def particles_video(particles, t1, shape = (256,256), t2 = None, 
                 background = 0, intensity = 10, sigma = None, dtype = "uint8"):
    """Creates brownian particles video
    
    Parameters
    ----------
    particles : iterable
        Iterable of particle coordinates
    t1 : array-like
        Frame time
    shape : (int,int)
        Frame shape
    t2 : array-like, optional
        Second camera frame time, in case we are simulating dual camera video.
    background : int
        Background frame value
    intensity : int
        Peak Intensity of the particle.
    sigma : float
        Sigma of the gaussian spread function for the particle
    dtype : dtype
        Numpy dtype of frames, either uint8 or uint16
        
    Yields
    ------
    frames : tuple of ndarrays
        A single-frame or dual-frame images (ndarrays).
    """
    dtype = _inspect_frame_dtype(dtype,intensity)
        
    background = np.zeros(shape = shape,dtype = dtype) + background
    height, width = shape
    
        
    out1 = None
    out2 = None
    
    t1 = list(t1)
    t2 = list(t2) if t2 is not None else None
        
    def get_frame(data):
        im = background.copy()

        if sigma is None:
            _int = np.asarray(intensity, dtype)
            if _int.ndim == 0:
                _int = np.asarray([intensity] * len(data), dtype)
            im = draw_points(im, data, _int)
        else:
            _int = np.asarray(intensity, dtype)
            if _int.ndim == 0:
                _int = np.asarray([intensity] * len(data),dtype)
            _sigma = np.asarray(sigma,FDTYPE)
            if _sigma.ndim == 0:
                _sigma = np.asarray([sigma] * len(data),FDTYPE)
            
            im = draw_psf(im, data, _int, _sigma)
        return im
    
    
    for i,data in enumerate(particles):
    
        if i in t1[:1]:
            t1.pop(0)
            out1 = get_frame(data)   
            
        if t2 is not None and i in t2[:1]:
            t2.pop(0)
            out2 = get_frame(data)  
            
        if t2 is not None:
            if out1 is not None and out2 is not None:
                yield out1, out2
                out1, out2 = None, None
        else:
            if out1 is not None:
                yield out1,
                out1 = None     

def data_trigger(data, indices):
    """A generator that selects data from an iterator 
    at given unique 'trigger' indices
    
    Examples
    --------
    >>> data = range(10)
    >>> indices = [1,4,7]
    >>> [x for x in data_trigger(data, indices)]
    [1, 4, 7]
    
    """
    
    t = list(reversed(sorted(set(indices))))  #make all indices unique and sorted. 
    
    ii = t.pop()
    
    for i,d in enumerate(data):
        if i == ii:
            yield d
            try:
                ii = t.pop()
            except IndexError:
                #all done.
                break
            
def plot_random_walk(count = 5000, num_particles = 2, shape = (256,256)):
    """Brownian particles usage example. Track 2 particles"""
    import matplotlib.pyplot as plt 
    x = np.array([x for x in brownian_particles(count = count, num_particles = num_particles, shape = shape)])
        
    for i in range(num_particles): 
        
        # Plot the 2D trajectory.
        plt.plot(x[:,i,0],x[:,i,1])
        
        # Mark the start and end points.
        plt.plot(x[0,i,0],x[0,i,1], 'go')
        plt.text(x[0,i,0],x[0,i,1], str(i))
        plt.plot(x[-1,i,0], x[-1,i,1], 'ro')
        plt.text(x[-1,i,0], x[-1,i,1], str(i))
        
    
    # More plot decorations.
    plt.title('2D Brownian Motion with mirror boundary conditions.')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.xlim(0,shape[1])
    plt.ylim(0,shape[0])
    plt.axis('equal')
    plt.grid(True)
    
def create_random_time(nframes, n = 32, dt_min = 1):
    """Create trigger time for single-camera random ddm experiments"""
    t0 = np.arange(nframes)*n
    r1 = 0
    for i in range(nframes):
        r1 = np.random.randint(max(0,dt_min-n+r1), n)
        t0[i] += r1
    return t0  
 
def random_time_count(nframes, n = 32):
    """Returns estimated count for single-camera random triggering scheme"""
    count = (np.arange(nframes*n+1)[::-1])/n/n
    count[0:n] = np.arange(n*1.)/(n) * count[0:n]
    count[0] = nframes * n
    return count
 
def create_random_times1(nframes,n = 32):
    """Create trigger times for c-ddm experiments based on Eq.7 from the paper"""
    iperiod = n * 2
    t0 = np.arange(nframes)*iperiod
    r1 = np.random.randint(n,  size = nframes)*(iperiod//(2*n))
    r2 = np.random.randint(2,  size = nframes)*(iperiod//(2))
    t1 = t0 +r1
    t2 = t0 + r2
    return t1, t2  

def create_random_times2(nframes,n = 32):
    """Create trigger times for c-ddm experiments based on Eq.8 from the paper"""
    iperiod = n * 2
    t0 = np.arange(nframes)*iperiod
    r1 = np.random.randint(n,  size = nframes)*(iperiod//(2*n))
    r2 = np.random.randint(2,  size = nframes)*(iperiod//(2))
    t1 = t0 +r1
    t2 = t0 + r2
    mask = np.random.randint(2*n,  size = nframes) == 0
    t2[mask] = t1[mask]
    return t1, t2 

def simple_brownian_video(t1, t2 = None, shape = (256,256), background = 0, intensity = 5, sigma = 3, dtype = "uint8", **kw):
    """DDM or c-DDM video generator.
 
    Parameters
    ----------
    t1 : array-like
        Frame time
    t2 : array-like, optional
        Second camera frame time, in case we are simulating dual camera video.
    shape : (int,int)
        Frame shape
    background : int
        Background frame value
    intensity : int
        Peak Intensity of the particle.
    sigma : float
        Sigma of the gaussian spread function for the particle
    dtype : dtype
        Numpy dtype of frames, either uint8 or uint16
    kw : extra arguments
        Extra keyward arguments that are passed to :func:`brownian_particles`
        
    Yields
    ------
    frames : tuple of ndarrays
        A single-frame or dual-frame images (ndarrays).
    
    """
    t1 = np.asarray(t1)
    if t2 is None:
        count = t1.max()+1
    else:
        t2 = np.asarray(t2)
        count = max(t1.max(),t2.max())+1
    kw["count"] = count 
    kw["shape"] = shape
    p = brownian_particles(**kw) 
    return particles_video(p, t1 = t1, t2 = t2, shape = shape, sigma = sigma, background = background,intensity = intensity, dtype = dtype) 

    
def adc(frame,  saturation = 32768, black_level = 0, bit_depth = "14bit", 
        readout_noise = 0., noise_model = "gaussian", out = None):
    """Simulated ADC conversion process of ideal signal.
    
    It applies shot noise with the standard deviation of the square of the
    provided signal and adds a readout_noise of the provided mean value and 
    shot noise characteristics.
    
    Parameters
    ----------
    frame : ndarray
        Input noisless signal
    saturation : int
        Defines the saturation value of the sensor
    black_level : int
        Defines black level subtraction value. 
    bit_depth : str
        ADC bit depth. Either '8bit', '10bit', '12bit' or '14bit'. This defines
        what kind of scalling is performed when converting results to uint16
        (or uint8) image. 
    readout_noise : float
        Value of the additional noise added to the frame.
    noise_model : str
        Either 'gaussian' or 'poisson' or 'none' to disable noise
    out : ndarray, optional
        Output array.
        
    Returns
    -------
    frame : ndarray
        Noissy image
    """
    readout_noise = np.array(readout_noise, FDTYPE)
    if noise_model == "gaussian":
        frame = shot_noise_gaussian(frame, readout_noise)
    elif noise_model == "poisson":
        frame = shot_noise_poisson(frame, readout_noise)
    elif noise_model != "none":
        raise ValueError("Invalid noise_model")
    max_value = 2**int(bit_depth[:-3]) - 1
    if max_value < 256:
        max_value = np.uint8(max_value)
    else:
        max_value = np.uint16(max_value)
    return adc_clean(frame,saturation,black_level, max_value, out = out)
        


        
#
#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()
#    seed(0)
#    plot_random_walk(1024,6)
