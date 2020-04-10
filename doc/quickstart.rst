.. _quickstart:

Quickstart Guide
================

This is a step-by-step example showing most typical use case of this package, that is, 
computation of the auto-correlation function for standard DDM experiments, and computation
of the cross-correlation function of cross-DDM experiments using multiple-tau algorithm
and real-time analysis.

For this guide you do not need the actual data because we will construct a simple
video of particles undergoing 2D Brownian motion using the (included in the package) Brownian motion simulator. In real world experiments you will have to stream
the videos either directly from the cameras with the library of choice, or read from the disk. This package does not provide tools for reading the data, so it is up to the user to make this task. You can use imageio_ or any other tool for loading recorded videos.

Video processing
----------------

Here we will cover some basic video processing functions and data types used in the package.

Video data
++++++++++

Video processing functions were designed to work on iterables of multi-frame data. Video data is an iterable object that consists of series of tuples, where each tuple holds a single numpy array (for single camera experiments) or a dual frame data (two numpy arrays) representing a cross-ddm experiment using two cameras. To clarify, a valid dual-frame  out-of-memory video can be constructed using python generators, e.g. for random noise dual_frame video you can do: 

.. doctest::

   >>> video = ((np.random.rand(512,512), np.random.rand(512,512)) for i in range(1024))

Here, video is a valid video iterable object that can be used to acquire frames frame-by-frame without needing the data to be read fully into memory, making it possible to analyze long videos. A valid single-frame video is therefore:

.. doctest::

   >>> video = ((np.random.rand(512,512),) for i in range(1024))

Notice the trailing comma to indicate a tuple of length 1. The above video is a multi-frame video holding only a single per element of the `video` iterable object. 

Showing video
+++++++++++++

For testing we will use a simple simulated video of a Brownian motion of 
100 particles. Here we will construct a simulation area of shape (512+32,512+32)
because we will later crop this to (512,512) to avoid mirroring of particles as they
touch the boundary of the simulation size because of the mirror boundary conditions implied by the simulation procedure. This way we simulate the real-world boundary effects better.

.. doctest::

   >>> from cddm.sim import simple_brownian_video
   >>> t = range(1024) #times at which to trigger the camera.
   >>> video = simple_brownian_video(t, shape = (512+32,512+32))
 
Here we have created a frame iterator of a video of a Brownian motion of spherical particles viewed with a camera triggered with a constant frame rate. Time `t` goes in units of time step, specified by the simulator. To load this into memory we can simply do.

.. doctest::
 
   >>> video = list(video)
   >>> video = tuple(video) #or this

.. note::

   For playin the video you are not required to load the data to memory. By doing so, it allows us to inspect the video back and forth, otherwise we can only iterate step by step in the forward direction with the :class:`.viewer.VideoViewer`.

Now we can inspect the video with

.. doctest::
 
   >>> from cddm.viewer import VideoViewer
   >>> viewer = VideoViewer(video, count = 1024)
   >>> viewer.show()

.. plot:: examples/show_video.py

   :class:`.viewer.VideoViewer` can be used to visualize the video (in memory or out-of-memory).

Cropping
++++++++

Sometimes you may want to crop data before processing. Cropping is done using pythons slice objects or simply, by specifying the range of values for slicing. For instance to perform slicing of frames of type ``ndarray[0:512,0:512]`` simply do:

.. doctest::
 
   >>> from cddm.video import crop
   >>> video = crop(video, roi = ((0,512), (0, 512)))

Under the hood, the crop function performs array slicing using slice object generated from the provided roi values. See :func:`.video.crop` for details.

Windowing
+++++++++

A common thing in FFT processing is to apply a window function to the data before we apply FFT. in :mod:`.window` there are several 2D windowing functions that you can use. After you have cropped the data you can apply window. First create window with the shape of our
frames shape (512,512). Remember, we have already cropped our original data to shape of (512,512)

.. doctest::
 
   >>> from cddm.window import blackman
   >>> window = blackman((512,512))

In order to multiply each frame of our video with this window function we must create another video-like object, that has the same length and frame shape as the video we wish to process.

.. doctest::
 
   >>> window_video = ((window,),)* 1024
   >>> video = multiply(video, window_video)

Optionally, if there are issues with the stability of the intensity of the light source you are using in the experiment, you can normalize each frame with respect to the mean value of the frame. This way you can avoid flickering effects, but you will introduce additional noise because of the randomness of the scattering process (randomness of the scattering intensity). 

.. doctest::
 
   >>> from cddm.video import normalize_video
   >>> video = normalize_video(video)

Performing FFT
++++++++++++++

Next thing is to compute the FFT of each frame in the video and to generate a FFT video.
This is again an iterable with multi-frame data, where each of the frames in the elements of the iterable correspond to the FFT of the original frames of the video that we are processing. Since input signal is real, there is no real benefit in using the general FFT algorithm for complex data and to hold reference to all computed Fourier coefficients, but it makes sense to compute only first half of the coefficients using np.fft.rfft2, for instance. 

Secondly, usually in DDM experiments there is a cutoff wavenumber value above which there is no measurable signal. To reduce memory requirements and computational effort it is therefore better to simply remove the data elements that are not needed. You can do this using

.. doctest::
 
   >>> from cddm.fft import rfft2
   >>> fft = rfft2(video, kimax = 31, kjmax = 31)

Here the resulting fft object is again of the same video data type. We have used two arguments `kimax` and `kjmax` for slicing. The result of this cropping is a video of FFTs, where the shape of each frame (in our case it is a single frame of the multi-frame data type) is (2*kimax+1, kjmax +1). As in uncropped rfft2 function, the zero wavenumber is at element [0,0], element [31,31] are for the wavenumber k = (31,31), element [-31,0] == [62,0] of the cropped fft is the Fourier coefficient of k = (-31,0). 

Optionally, you can normalize for flickering effects in fft space, instead of normalization performed in real space.

.. doctest::
 
   >>> from cddm.fft import normalize_fft
   >>> fft = normalize_fft(fft)

Again, do this only if you have problems with the stability of the light source.

Bakground removal
+++++++++++++++++

It is important that background removal is performed at some stage, either before the computation of the correlation or after, using proper normalization procedure. If you can obtain the (possibly time-dependent) background frame from a separate experiment you can subtract the frames either in real space (done before calling rfft2):

.. doctest::

   >>> background = np.zeros((512,512))
   >>> background_video = ((background,),) * 1024
   >>> video = subtract(video, background_video)

or better, in reciprocal space:

.. doctest::

   >>> background = np.zeros((63,32)) + 0j 
   >>> background_fft = ((background,),) * 1024
   >>> fft = subtract(fft, background_fft)

However, most of the time it is not possible to acquire a good estimator of the background image, so in the correlation calculations we will rely on a proper normalization procedure.

Of course none of the processing has yet take place till this stage because all processing functions that were applied have not yet been executed. The execution of the video processing function takes place in real-time when we calculate the correlation function. If you do need to inspect the results of the processing you have to load the calcualtion results in memory.

To load the results of the processing into memory, to inspect the data you can do

.. doctest::

   >>> fft = list(fft)
   >>> fft = tuple(fft) #or this

.. note::

   You do not need to load the data into memory. The calculation of the correlation function using multiple tau algorithm does not require all data to be read at once, so you should generally not load the data to memory. Also, for real-time calculations and on-the-fly correlation data visualization you should not load the fft data into memory!

Auto-correlation
----------------

Now that our video has been cropped, windowed, normalized, fourier transformed, we can start calculating the correlation function. There a few ways how to calculate the correlation function (or image structure function) with the `cddm` package. Here we will only cover the multiple-tau approach, as this is the most efficient way to simultaneously obtain small tau and large tau data. There is an in-memory version of the algorithm, working on numpy arrays and an out-of-memory version working on video data iterable object, as explained above. Here we will cover the out-of-memory approach. For the in-memory version and examples browse through the examples in the source.

Calculation
+++++++++++

To perform correlation analysis you have to provide the FFT iterator and time sequence identifieng the time in unit step at which the frame was captured. In our case, for standard DDM this is simple a range of integers of length matching the video length.

.. doctest::

   >>> from cddm.multitau import iacorr_multi
   >>> data, bg, var = iacorr_multi(fft, t)

The output of the :func:`.multitau.iacorr_multi`, by default, returns a data tuple with a structure that will be defined shortly, and two additional arrays (mean pixel value array and pixel variance) that are needed for normalization. First, let us inspect the data using :class:`.viewer.MultitauViewer`

.. doctest::
   
   >>> from cddm.viewer import MultitauViewer
   >>> viewer = MultitauViewer(scale = True)
   >>> viewer.set_data(data, bg, var)
   >>> viewer.set_mask(k = 25, angle = 0, sector = 30)
   True
   >>> viewer.plot()
   >>> viewer.show()

We used the `scale = True` option to normalize data to pixel variance, which results in scaling the data between (0,1). 

.. plot:: examples/auto_correlate_multi_live.py

   :class:`.viewer.MultitauViewer` can be used to visualize the correlation data. With sliders you can select the size of the wave vector `k`, angle of the wave vector with respect to the horizontal axis, and averaging sector. The resulting correlation function that is shown on the left subplot is a mean value of the computed correlation functions at the wave vectors that are marked in the right subplot.

Data structure
++++++++++++++

The multitau correlation data itself resides in a tuple of two elements

.. doctest::
 
   >>> lin_data, multi_level = data

Both `lin_data` and `multi_data` are a tuple of numpy arrays. The actual correlation data is the first element

.. doctest::

   >>> corr_lin = lin_data[0]
   >>> corr_multi = multi_level[0]

The second element is count data, needed for the most basic normalization

.. doctest::

   >>> count_lin = lin_data[1]
   >>> count_multi = multi_level[1]

Here the shape of the data are

.. doctest::

   >>> corr_lin.shape == (63,32,16) and count_lin.shape == (16,)
   True
   >>> corr_multi.shape == (6,63,32,16) and count_multi.shape == (6,16)
   True

By default the size of each level in multilevel data is 16, so we have 16 time delays for each level, and there are 63 times 32 unique k values. The multi_level part of the data has 5 levels, the length of corr_multi varies, and depends on the length of the video. 

Normalization
+++++++++++++

Normally, you won't work with raw correlation data and you will perform normalization using:

.. doctest::

   >>> from cddm.multitau import normalize_multi, log_merge
   >>> lin_data, multi_level = normalize_multi(data, bg, var, scale = True)

Here, `lin_data` and `multi_level` are numpy arrays of normalized correlation data.  One final step is to merge the multi_level part with the linear part into one continuous log-spaced data.

.. doctest::

   >>> x, y = log_merge(lin_data, multi_level)

Here, `x` is a time delay array, `y` is the merged correlation data. The first two axes are for the i- and j-indices of the wave vector k = (ki,kj). So to plot the computed correlation function as a function of time for a few wave vectors, for instance:

.. doctest::

   >>> import matplotlib.pyplot as plt
   >>> for (i,j) in ((4,12),(-6,16), (6,16)):
   ...     ax = plt.semilogx(x,y[i,j], label =  "k = ({}, {})".format(i,j))
   >>> legend = plt.legend()
   >>> text = plt.xlabel("time delay")
   >>> text = plt.ylabel("G/Var")
   >>> plt.show()

.. plot:: examples/auto_correlate_multi_data_plot.py

   Data was normalized and scaled, so the computed correlation is limited between (0,1). 


Cross-correlation
-----------------

For cross correlation on randomly-triggered dual-camera system, as demonstrated in the paper_, the computation is basically the same. Cross-correlation with irregular spaced data can be done in the following way. Import the tools needed:

.. doctest::

   >>> from cddm.viewer import MultitauViewer
   >>> from cddm.video import multiply,  crop
   >>> from cddm.window import blackman
   >>> from cddm.fft import rfft2
   >>> from cddm.multitau import iccorr_multi, normalize_multi, log_merge
   >>> from cddm.sim import simple_brownian_video, create_random_times1

Now set up random time sequence and video of cross-DDM 

.. doctest::

   >>> t1, t2 = create_random_times1(1024,n = 16)
   >>> video = simple_brownian_video(t1,t2, shape = (512+32,512+32))
   >>> video = crop(video, roi = ((0,512), (0,512)))

We will apply some dust particles to each frame in order to simulate different static background 
on the two cameras. If you working directory is in the `examples` folder you can load dust images::

   >>> dust1 = plt.imread('dust1.png')[...,0] #float normalized to (0,1)
   >>> dust2 = plt.imread('dust2.png')[...,0]
   >>> dust = ((dust1,dust2),)*nframes
   >>> video = multiply(video, dust)

To view the two videos we can again use the VideoViewer

.. doctest::

   >>> video = list(video) 
   >>> viewer1 = VideoViewer(video, count = 1024, id = 0)
   >>> viewer1.show()
   >>> viewer2 = VideoViewer(video, count = 1024, id = 1)
   >>> viewer2.show()

.. plot:: examples/show_dual_video.py

   Dust particles on the two cameras are different, which result in different background frames.


Pre-process video and perform FFT

.. doctest::

   >>> window = blackman((512,512))
   >>> window_video = ((window,window),)*1024
   >>> video = multiply(video, window_video)
   >>> fft = rfft2(video, kimax =37, kjmax = 37)

Live view
+++++++++

To show live view of the computed correlation function, we can pass the viewer as an argument to :func:`.multitau.iccorr_multi`:

.. doctest:: 
   
   >>> viewer = MultitauViewer(scale = True)
   >>> viewer.k = 15 #initial mask parameters
   >>> viewer.sector = 30
   >>> data, bg, var = iccorr_multi(fft, t1, t2, period = 32, viewer  = viewer)

Note the `period` argument. You must provide the correct effective period of the random triggering of the cross-ddm experiment. Otherwise, data will not be merged and processed correctly. Care must be taken not to mix up this parameter, as there is no easy way to determine the period from t1, and t2 parameters alone. The `bg` and `var` are now tuples of arrays of mean pixel and pixel variances of each of the two videos.


Normalization options
---------------------

One important note is on the normalization flags that you can use and the `method` option in the :func:`.multitau.icorr_multi`, :func:`.multitau.iacorr_multi`. By default, computation and normalization is performed using

.. doctest:: 

   >>> from cddm.core import NORM_COMPENSATED, NORM_SUBTRACTED, NORM_BASELINE
   >>> norm = NORM_COMPENSATED | NORM_SUBTRACTED
   >>> norm == 3
   True

This way it is possible to normalize the computed data in four different ways.

.. doctest:: 
   
   >>> data_0 = normalize_multi(data, bg, var, norm = NORM_BASELINE, scale = True) #norm = 0
   >>> x_0, y_0 = log_merge(*data_0)
   >>> data_1 = normalize_multi(data, bg, var, norm = NORM_COMPENSATED, scale = True) #norm = 1
   >>> x_1, y_1 = log_merge(*data_1)
   >>> data_2 = normalize_multi(data, bg, var, norm = NORM_SUBTRACTED, scale = True) #norm = 2
   >>> x_2, y_2 = log_merge(*data_2)
   >>> data_3 = normalize_multi(data, bg, var, norm = NORM_COMPENSATED|NORM_SUBTRACTED, scale = True) #norm = 3
   >>> x_3, y_3 = log_merge(*data_3)
   >>> i,j = 4,15
   >>> ax = plt.semilogx(x_0,y_0[i,j], label =  "norm = 0" )
   >>> ax = plt.semilogx(x_1,y_1[i,j], label =  "norm = 1" )
   >>> ax = plt.semilogx(x_2,y_2[i,j], label =  "norm = 2" )
   >>> ax = plt.semilogx(x_3,y_3[i,j], label =  "norm = 3" )
   >>> text = plt.xlabel("t")
   >>> text = plt.ylabel("G / Var")
   >>> legend = plt.legend()
   >>> plt.show()

.. plot:: examples/cross_correlate_multi_norm_plot.py

   Normalization mode 2 or 3 work best, but require more intense computations.

If you decide from the start which normalization mode are you going to use, you can set this mode 


.. _imageio: https://github.com/imageio/imageio
.. _paper: https://doi.org/10.1039/C9SM00121B
