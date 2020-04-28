:mod:`cddm.sim`
===============

.. py:module:: cddm.sim

.. autoapi-nested-parse::

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



Module Contents
---------------

.. function:: seed(value)

   Seed for numba and numpy random generator


.. function:: brownian_walk(x0, count=1024, shape=(256, 256), delta=1, dt=1, velocity=(0.0, 0.0))

   Returns an brownian walk iterator.

   Given the initial coordinates x0, it callculates and yields next `count` coordinates.

   :param x0: A list of initial coordinates (i, j) of particles (in pixel units).
   :type x0: array-like
   :param count: Number of simulation steps.
   :type count: int
   :param shape: Shape of the simulation region in pixels.
   :type shape: (int,int)
   :param delta: Defines an average step in pixel coordinates (when dt = 1).
   :type delta: float
   :param dt: Simulation time step (1 by default).
   :type dt: float, optional
   :param velocity: Defines an average velocity (vi,vj) in pixel coordinates per unit time step
                    (when dt = 1).
   :type velocity: (float,float), optional

   :Yields: **coordinates** (*ndarray*) -- Coordinates 2D array for the particles. The second axis is the x,y coordinate.


.. function:: brownian_particles(count=500, shape=(256, 256), particles=100, delta=1, dt=1, velocity=0.0, x0=None)

   Coordinates generator of multiple brownian particles.

   Builds particles randomly distributed in the computation box and performs
   random walk of coordinates.

   :param count: Number of steps to calculate
   :type count: int
   :param shape: Shape of the box
   :type shape: (int,int)
   :param particles: Number of particles in the box
   :type particles: int
   :param delta: Step variance in pixel units (when dt = 1)
   :type delta: float
   :param dt: Time resolution
   :type dt: float
   :param velocity: Velocity in pixel units (when dt = 1)
   :type velocity: float
   :param x0: A list of initial coordinates
   :type x0: array-like

   :Yields: **coordinates** (*ndarray*) -- Coordinates 2D array for the particles. The second axis is the x,y coordinate.
            Length of the array equals number of particles.


.. function:: particles_video(particles, t1, shape=(256, 256), t2=None, background=0, intensity=10, sigma=None, noise=0.0)

   Creates brownian particles video

   :param particles: Iterable of particle coordinates
   :type particles: iterable
   :param t1: Frame time
   :type t1: array-like
   :param shape: Frame shape
   :type shape: (int,int)
   :param t2: Second camera frame time, in case we are simulating dual camera video.
   :type t2: array-like, optional
   :param background: Background frame value
   :type background: int
   :param intensity: Peak Intensity of the particle.
   :type intensity: int
   :param sigma: Sigma of the gaussian spread function for the particle
   :type sigma: float
   :param noise: Intensity of the random noise
   :type noise: float, optional

   :Yields: **frames** (*tuple of ndarrays*) -- A single-frame or dual-frame images (ndarrays).


.. function:: data_trigger(data, indices)

   A generator that selects data from an iterator
   at given unique 'trigger' indices

   .. rubric:: Examples

   >>> data = range(10)
   >>> indices = [1,4,7]
   >>> [x for x in data_trigger(data, indices)]
   [1, 4, 7]


.. function:: plot_random_walk(count=5000, particles=2, shape=(256, 256))

   Brownian particles usage example. Track 2 particles


.. function:: create_random_times1(nframes, n=20)

   Create trigger times for c-ddm experiments based on Eq.7 from the paper


.. function:: create_random_times2(nframes, n=20)

   Create trigger times for c-ddm experiments based on Eq.8 from the paper


.. function:: simple_brownian_video(t1, t2=None, shape=(256, 256), background=0, intensity=5, sigma=3, noise=0, **kw)

   DDM or c-DDM video generator.


   :param t1: Frame time
   :type t1: array-like
   :param t2: Second camera frame time, in case we are simulating dual camera video.
   :type t2: array-like, optional
   :param shape: Frame shape
   :type shape: (int,int)
   :param background: Background frame value
   :type background: int
   :param intensity: Peak Intensity of the particle.
   :type intensity: int
   :param sigma: Sigma of the gaussian spread function for the particle
   :type sigma: float
   :param noise: Intensity of the random noise
   :type noise: float, optional
   :param kw: Extra keyward arguments that are passed to :func:`brownian_particles`
   :type kw: extra arguments

   :Yields: **frames** (*tuple of ndarrays*) -- A single-frame or dual-frame images (ndarrays).


