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
import numba as nb
import math

from cddm.conf import U8,F,I,FDTYPE, NUMBA_TARGET, NUMBA_PARALLEL , NUMBA_CACHE, NUMBA_FASTMATH

@nb.vectorize([F(F,F,F)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
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

def seed(value):
    """Seed for numba and numpy random generator"""
    np.random.seed(value)
    numba_seed(value)

@nb.njit()
def numba_seed(value):
    """Seed for numba random generator"""
    if NUMBA_TARGET != "cpu":
        print("WARNING: Numba seed not effective. Seeding only works with NUMBA_TARGET = 'cpu'")
    np.random.seed(value)
                          
@nb.vectorize([F(F,F,F)], target = NUMBA_TARGET, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)        
def make_step(x,scale, velocity):
    """Performs random particle step from a given initial position x."""
    return x + np.random.randn()*scale + velocity   


def brownian_walk(x0, count = 1024, shape = (256,256), delta = 1, dt = 1, velocity = 0.):
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
    dt : float
        Simulation time step.
    velocity : (float,float)
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
        
def brownian_particles(count = 500, shape = (256,256),particles = 100, delta = 1, dt = 1,velocity = 0., x0 = None):
    """Coordinates generator of multiple brownian particles.
    
    Builds particles randomly distributed in the computation box and performs
    random walk of coordinates.
    
    Parameters
    ----------
    count : int
        Number of steps to calculate
    shape : (int,int)
        Shape of the box
    particles : int
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
        x0 = np.asarray(np.random.rand(int(particles),2)*np.array(shape),FDTYPE)
    else:
        x0 = np.asarray(x0,FDTYPE)
        if x0.ndim == 1 and particles == 1:
            x0 = x0[None,:]
        if len(x0) != particles:
            raise ValueError("Wrong length of initial coordinates x0")
 
    v0 = np.zeros_like(x0)
    v0[:,:] = velocity
    for data in brownian_walk(x0,count,shape,delta,dt,v0):
        yield data
             
@nb.jit([U8(I,F,I,F,F,U8)],nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)
def psf_gauss(x,x0,y,y0,sigma,intensity):
    """Gaussian point-spread function. This is used to calculate pixel value
    for a given pixel coordinate x,y and particle position x0,y0."""
    return intensity*math.exp(-0.5*((x-x0)**2+(y-y0)**2)/(sigma**2))
                 
@nb.jit(["uint8[:,:],float64[:,:],uint8[:]"], nopython = True, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                
def draw_points(im, points, intensity):
    """Draws pixels to image from a given points array"""
    data = points
    nparticles = len(data)
    assert points.shape[1] == 2
    assert nparticles == len(intensity)
    for j in range(nparticles):
        im[int(data[j,0]),int(data[j,1])] = im[int(data[j,0]),int(data[j,1])] + intensity[j] 
    return im   
        
@nb.jit([U8[:,:](U8[:,:],F[:,:],U8[:],F[:])], nopython = True, parallel = NUMBA_PARALLEL, cache = NUMBA_CACHE, fastmath = NUMBA_FASTMATH)                
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

def particles_video(particles, t1, shape = (256,256), t2 = None, 
                 background = 0, intensity = 10, sigma = None, noise = 0.):
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
    noise : float, optional
        Intensity of the random noise
        
    Yields
    ------
    frames : tuple of ndarrays
        A single-frame or dual-frame images (ndarrays).
    """
        
    background = np.zeros(shape = shape,dtype = "uint8") + background
    height, width = shape
    
        
    out1 = None
    out2 = None
    
    t1 = list(t1)
    t2 = list(t2) if t2 is not None else None
        
    def get_frame(data, noise):
        im = background.copy()
        if noise > 0.:
            n = np.uint8(np.random.poisson(noise, shape))
            im = np.add(im, n, im)

        if sigma is None:
            _int = np.asarray(intensity, "uint8")
            if _int.ndim == 0:
                _int = np.asarray([intensity] * len(data), "uint8")
            im = draw_points(im, data, _int)
        else:
            _int = np.asarray(intensity, "uint8")
            if _int.ndim == 0:
                _int = np.asarray([intensity] * len(data), "uint8")
            _sigma = np.asarray(sigma,FDTYPE)
            if _sigma.ndim == 0:
                _sigma = np.asarray([sigma] * len(data),FDTYPE)
            
            im = draw_psf(im, data, _int, _sigma)
        return im
    
    
    for i,data in enumerate(particles):
    
        if i in t1[:1]:
            t1.pop(0)
            out1 = get_frame(data,noise)   
            
        if t2 is not None and i in t2[:1]:
            t2.pop(0)
            out2 = get_frame(data,noise)  
            
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
            
def plot_random_walk(count = 5000, particles = 2, shape = (256,256)):
    """Brownian particles usage example. Track 2 particles"""
    import matplotlib.pyplot as plt 
    x = np.array([x for x in brownian_particles(count = count, particles = particles, shape = shape)])
        
    for i in range(particles): 
        
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
 
def create_random_times1(nframes,n = 20):
    """Create trigger times for c-ddm experiments based on Eq.7 from the paper"""
    iperiod = n * 2
    t0 = np.arange(nframes)*iperiod
    r1 = np.random.randint(n,  size = nframes)*(iperiod//(2*n))
    r2 = np.random.randint(2,  size = nframes)*(iperiod//(2))
    t1 = t0 +r1
    t2 = t0 + r2
    return t1, t2  

def create_random_times2(nframes,n = 20):
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

def simple_brownian_video(t1, t2 = None, shape = (256,256), background = 0, intensity = 5, sigma = 3, noise = 0, **kw):
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
    noise : float, optional
        Intensity of the random noise
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
    return particles_video(p, t1 = t1, t2 = t2, shape = shape, sigma = sigma, background = background,intensity = intensity, noise = noise) 


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    seed(0)
    plot_random_walk(1024,6)
