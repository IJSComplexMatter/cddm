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

.. function:: form_factor(window, sigma=3, intensity=5, navg=10, dtype='uint8', mode='rfft2')

   Computes point spread function form factor.

   Draws a PSF randomly in the frame and computes FFT, returns average absolute
   of the FFT.

   :param window: A 2D window function used in the analysis. Set this to np.ones if you do
                  not use one.
   :type window: ndarray
   :param sigma: Sigma of the PSF of the image
   :type sigma: float
   :param intensity: Intensity value
   :type intensity: unsigned int
   :param navg: Specifies mesh size for averaging.
   :type navg: int
   :param dtype: One of "uint8" or "uint16". Defines output dtype of the image.
   :type dtype: np.dtype
   :param mode: Either 'rfft2' (default) or 'fft2'.
   :type mode: str, optional

   :returns: **out** -- Average absolute value of the FFT of the PSF.
   :rtype: ndarray


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


.. function:: brownian_particles(count=500, shape=(256, 256), num_particles=100, delta=1, dt=1, velocity=0.0, x0=None)

   Coordinates generator of multiple brownian particles.

   Builds particles randomly distributed in the computation box and performs
   random walk of coordinates.

   :param count: Number of steps to calculate
   :type count: int
   :param shape: Shape of the box
   :type shape: (int,int)
   :param num_particles: Number of particles in the box
   :type num_particles: int
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


.. function:: particles_video(particles, t1, shape=(256, 256), t2=None, background=0, intensity=10, sigma=None, dtype='uint8')

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
   :param dtype: Numpy dtype of frames, either uint8 or uint16
   :type dtype: dtype

   :Yields: **frames** (*tuple of ndarrays*) -- A single-frame or dual-frame images (ndarrays).


.. function:: data_trigger(data, indices)

   A generator that selects data from an iterator
   at given unique 'trigger' indices

   .. rubric:: Examples

   >>> data = range(10)
   >>> indices = [1,4,7]
   >>> [x for x in data_trigger(data, indices)]
   [1, 4, 7]


.. function:: plot_random_walk(count=5000, num_particles=2, shape=(256, 256))

   Brownian particles usage example. Track 2 particles


.. function:: create_random_time(nframes, n=32, dt_min=1)

   Create trigger time for single-camera random ddm experiments


.. function:: random_time_count(nframes, n=32)

   Returns estimated count for single-camera random triggering scheme


.. function:: create_random_times1(nframes, n=32)

   Create trigger times for c-ddm experiments based on Eq.7 from the paper


.. function:: create_random_times2(nframes, n=32)

   Create trigger times for c-ddm experiments based on Eq.8 from the paper


.. function:: simple_brownian_video(t1, t2=None, shape=(256, 256), background=0, intensity=5, sigma=3, dtype='uint8', **kw)

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
   :param dtype: Numpy dtype of frames, either uint8 or uint16
   :type dtype: dtype
   :param kw: Extra keyward arguments that are passed to :func:`brownian_particles`
   :type kw: extra arguments

   :Yields: **frames** (*tuple of ndarrays*) -- A single-frame or dual-frame images (ndarrays).


.. function:: adc(frame, saturation=32768, black_level=0, bit_depth='14bit', readout_noise=0.0, noise_model='gaussian', out=None)

   Simulated ADC conversion process of ideal signal.

   It applies shot noise with the standard deviation of the square of the
   provided signal and adds a readout_noise of the provided mean value and
   shot noise characteristics.

   :param frame: Input noisless signal
   :type frame: ndarray
   :param saturation: Defines the saturation value of the sensor
   :type saturation: int
   :param black_level: Defines black level subtraction value.
   :type black_level: int
   :param bit_depth: ADC bit depth. Either '8bit', '10bit', '12bit' or '14bit'. This defines
                     what kind of scalling is performed when converting results to uint16
                     (or uint8) image.
   :type bit_depth: str
   :param readout_noise: Value of the additional noise added to the frame.
   :type readout_noise: float
   :param noise_model: Either 'gaussian' or 'poisson' or 'none' to disable noise
   :type noise_model: str
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **frame** -- Noissy image
   :rtype: ndarray


