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

.. function:: mirror(x, x0, x1)

   transforms coordinate x by flooring in the interval of [x0,x1]
   It performs x0 + (x-x0)%(x1-x0)


.. function:: seed(value)

   Seed for numba and numpy random generator


.. function:: numba_seed(value)

   Seed for numba random generator


.. function:: make_step(x, scale, velocity)

   Performs random particle step from a given initial position x.


.. function:: brownian_walk(x0, n=1024, shape=(256, 256), delta=1, dt=1, velocity=0.0)

   Returns an brownian walk iterator.

   Given the initial coordinates x0, it callculates and yields next n coordinates.


   :param x0: A list of initial coordinates (i, j) of particles (in pixel units)
   :type x0: array-like
   :param n: Number of simulation steps
   :type n: int
   :param shape: Shape of the simulation region in pixels
   :type shape: (int,int)
   :param delta: Defines an average step in pixel coordinates (when dt = 1).
   :type delta: float
   :param dt: Simulation time step.
   :type dt: float
   :param velocity: Defines an average velocity (vi,vj) in pixel coordinates per unit time step
                    (when dt = 1).
   :type velocity: (float,float)


.. function:: brownian_particles(n=500, shape=(256, 256), particles=100, delta=1, dt=1, velocity=0.0, x0=None)

   Creates coordinates of multiple brownian particles.

   :param n: Number of steps to calculate
   :type n: int
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


.. function:: psf_gauss(x, x0, y, y0, sigma, intensity)

   Gaussian point-spread function. This is used to calculate pixel value
   for a given pixel coordinate x,y and particle position x0,y0.


.. function:: draw_points(im, points, intensity)

   Draws pixels to image from a given points array


.. function:: draw_psf(im, points, intensity, sigma)

   Draws psf to image from a given points array


.. function:: particles_video(particles, t1, shape=(512, 512), t2=None, background=0, intensity=10, sigma=None, noise=0.0)

   Creates brownian particles video


.. function:: data_trigger(data, indices)

   A generator that selects data from an iterator
   at given unique 'trigger' indices

   .. rubric:: Examples

   >>> data = range(10)
   >>> indices = [1,4,7]
   >>> [x for x in data_trigger(data, indices)]
   [1, 4, 7]


.. function:: test_plot(n=5000, particles=2)

   Brownian particles usage example. Track 2 particles


.. function:: create_random_times1(nframes, n=20)

   Create trigger times for c-ddm experiments based on Eq.7 from the paper


.. function:: create_random_times2(nframes, n=20)

   Create trigger times for c-ddm experiments based on Eq.8 from the paper


