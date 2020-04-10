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

To proceed you can copy-paste the example below in your python shell. Or, open the source files under each of the plots in this document and run those.

Video processing
----------------

Here we will cover some basic video processing functions and data types used in the package. We will demonstrate the use on single-camera video first.

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


Performing FFT
++++++++++++++

Next thing is to compute the FFT of each frame in the video and to generate a FFT video.
This is again an iterable with multi-frame data, where each of the frames in the elements of the iterable correspond to the FFT of the original frames of the video that we are processing. Since input signal is real, there is no real benefit in using the general FFT algorithm for complex data and to hold reference to all computed Fourier coefficients, but it makes sense to compute only first half of the coefficients using np.fft.rfft2, for instance. 

Secondly, usually in DDM experiments there is a cutoff wavenumber value above which there is no measurable signal. To reduce memory requirements and computational effort it is therefore better to simply remove the data elements that are not needed. You can do this using

.. doctest::
 
   >>> from cddm.fft import rfft2
   >>> fft = rfft2(video, kimax = 31, kjmax = 31)

Here the resulting fft object is again of the same video data type. We have used two arguments `kimax` and `kjmax` for slicing. The result of this cropping is a video of FFTs, where the shape of each frame (in our case it is a single frame of the multi-frame data type) is (2*kimax+1, kjmax +1). As in uncropped rfft2 function, the zero wavenumber is at element [0,0], element [31,31] are for the wavenumber k = (31,31), element [-31,0] == [62,0] of the cropped fft is the Fourier coefficient of k = (-31,0). 


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

There are two or three extra arrays in the lin_data and multi_data tuples. What comes after the count data depends on the normalization procedure and the method that is used in the calculation of the correlation function. Different normalization procedures are implemented and there are different ways to calculate the correlation function. This will be covered in detail later. Normally, you won't work with raw correlation data and you will perform normalization using:

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

You are done, now you can save the data in numpy data format for later use::

   >>> np.save("t.npy", x)
   >>> np.save("data.npy", y)

If you wish to analyze the data with some other tool (Mathematica, Origin) you will have to google for help on how to import the numpy binary data. Another option is to save as text files. But you have to do it index by index. For instance, to save the (4,8) k-value data, you can do.

.. doctest::

   >>> i, j = 4, 8
   >>> np.savetxt("t.txt", x)
   >>> np.savetxt("data_{}_{}.txt".format(i,j), y[i,j])

Now you can use your favorite (python ;) ) tool for data analysis and fitting.

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

Light flickering
++++++++++++++++

In cross-DDM if using pulsed light source, and if there are issues with the stability of the intensity of the light source you are using in the experiment, you can normalize each frame with respect to the mean value of the frame. This way you can avoid flickering effects, but you will introduce additional noise because of the randomness of the scattering process (randomness of the scattering intensity). 

.. doctest::
 
   >>> from cddm.video import normalize_video
   >>> video = normalize_video(video)


Pre-process video and perform FFT

.. doctest::

   >>> window = blackman((512,512))
   >>> window_video = ((window,window),)*1024
   >>> video = multiply(video, window_video)
   >>> fft = rfft2(video, kimax =37, kjmax = 37)

Optionally, you can normalize for flickering effects in fft space, instead of normalization performed in real space.

.. doctest::
 
   >>> from cddm.fft import normalize_fft
   >>> fft = normalize_fft(fft)
   >>> fft = list(fft) #not really needed if you are going to process fft only once

Again, do this only if you have problems with the stability of the light source.

Live view
+++++++++

To show live view of the computed correlation function, we can pass the viewer as an argument to :func:`.multitau.iccorr_multi`:

.. doctest:: 
   
   >>> viewer = MultitauViewer(scale = True)
   >>> viewer.k = 15 #initial mask parameters
   >>> viewer.sector = 30
   >>> data, bg, var = iccorr_multi(fft, t1, t2, period = 32, viewer  = viewer)

.. plot:: examples/cross_correlate_multi_live.py

   You can see the computation in real-time.


Note the `period` argument. You must provide the correct effective period of the random triggering of the cross-ddm experiment. Otherwise, data will not be merged and processed correctly. Care must be taken not to mix up this parameter, as there is no easy way to determine the period from t1, and t2 parameters alone. The `bg` and `var` are now tuples of arrays of mean pixel and pixel variances of each of the two videos.


Normalization options
---------------------

One important note is on the normalization flags that you can use and the `method` option in the :func:`.multitau.iccorr_multi` and :func:`.multitau.iacorr_multi` . By default, computation and normalization is performed using

.. doctest:: 

   >>> from cddm.core import NORM_COMPENSATED, NORM_SUBTRACTED, NORM_BASELINE
   >>> norm = NORM_COMPENSATED | NORM_SUBTRACTED
   >>> norm == 3
   True

This way it is possible to normalize the computed data with the :func:`.multitau.normalize_multi` function in four different ways:

* *baseline* : norm = NORM_BASELINE (norm = 0), here we remove the baseline error introduced by the non-zero background fram, which produces an offset in the correlation data. For this to work, you must provide the background data to the :func:`.multitau.normalize_multi`
* *compensated* : norm = NORM_COMPENSATED (norm = 1), here we compensate the error introduced at smaller delay times, which is due to non-ergodicity of the data. Basically, we normalize the data as if we calculated the cross-difference function instead of the cross-correlation. This requires one to calculate the delay-dependent squares of the intensities, which slows down the computation
* *subtracted* : norm = NORM_SUBTRACTED (norm = 2), here we compensate for baseline error and for the linear error introduced by the not-known in advance background data. This requires one to track the delay-dependent sum of the data, which further slows down the computation
* *subtracted and compensated* : norm = NORM_COMPENSATED | NORM_SUBTRACTED (norm = 3), which does both the subtracted and compensated normalizations.

.. doctest:: 
   
   >>> i,j = 4,15
   >>> for norm in (0,1,2,3):
   ...    fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
   ...    x,y = log_merge(fast, slow)
   ...    ax = plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )
   >>> text = plt.xlabel("t")
   >>> text = plt.ylabel("G / Var")
   >>> legend = plt.legend()
   >>> plt.show()

.. plot:: examples/cross_correlate_multi_norm_plot.py

   Normalization mode 3 works best for small time delays, mode 2 works best for large delays and is more noisy at smaller delays.

If you decide from the start which normalization mode are you going to use, you can set the normalization mode. This may reduce the computational effort in some cases. For instance, the main reason to use modes 2 and 3 is to properly remove the two different background frames from both cameras. Usually, this background frame is not known until the experiment is finished, so the background subtraction is done after the calculation  of the correlation function is performed. However, this requires that we track two extra channels measuring the delay-dependent data sum for each of the camera, or one additional channel measureing the delay-dependent sum of the squares of the data on both cameras. This significantly slows down the computation.

One way to partially overcome this limitation is to use the `auto_background` option and to define a large enough `chunk_size`

.. doctest::

   >>> data, bg, var = iccorr_multi(fft, t1, t2, period = 32, chunk_size = 512, auto_background = True)

This way we have forced the algorithm to work with chunks of data of length 512, and to take the first chunk of data to calculate the background frames that are then used to subtract from the input video. This way we gat a reasonably good estimator of the background, which reduces the need to use the NORM_SUBTRACTED flag for the normalization as shown below.


.. doctest:: 
   
   >>> i,j = 4,15
   >>> for norm in (0,1,2,3):
   ...    fast, slow = normalize_multi(data, bg, var, norm = norm, scale = True)
   ...    x,y = log_merge(fast, slow)
   ...    ax = plt.semilogx(x,y[i,j], label =  "norm = {}".format(norm) )
   >>> text = plt.xlabel("t")
   >>> text = plt.ylabel("G / Var")
   >>> legend = plt.legend()
   >>> plt.show()

.. plot:: examples/cross_correlate_multi_subtracted.py

   Background frame has been succesfuly subtracted and there isn no real benefit in using the NORM_SUBTRACTED flag (norm = 2 or norm = 3), and we can work with NORM_BASELINE (norm = 0) or NORM_COMPENSATED (norm = 1).

.. note::
   
   If background is properly subtracted before the calculation of the correlation function, the output of  `normalize_multi` with norm = 0 and norm = 2 are identical, and the output of `normalize_multi` with norm = 1 and norm = 3 are identical. In the case above, background has not been fully subtracted, so there is still a small difference.

In some experiments, it may be sufficient to work with norm = 0, and you can instead work with::

   >>> data, bg, var = iccorr_multi(fft, t1, t2, period = 32, 
   ...         norm = NORM_BASELINE, chunk_size = 512, auto_background = True)

which will significantly improve the speed of computation, as there is no need to track the three extra channels. In case you do need the `compensated` normalization, you can do:

   >>> data, bg, var = iccorr_multi(fft, t1, t2, period = 32, 
   ...         norm = NORM_COMPENSATED, chunk_size = 512, auto_background = True)

This will allow you to normalize other to `baseline` or `compensated`, but the computation is slower because of the two extra channels that need to be calculated.

.. note::

   In non-ergodic systems auto-background subtraction may not be good enough, so you are encouraged to work with norm = 3 (the default) during the calculation, and later decide on the normalization procedure. You should calculate with norm < 3 only if you need to gain the speed, or reduce the memory requirements.

Data analysis
-------------

Now that we have calculated the correlation function, it is time to do one final step: we need to analyze the data. First, to improve the statistics, it is wise to perform some sort of k-averaging over neighboring wave vectors. We have already use the MultitauViewer to visualize the data and do the averaging, so we can use the viewer to obtain the k-averaged data:

.. doctest:: 

   >>> ok = viewer.set_mask(k = 10, angle = 0, sector = 30)
   >>> if ok: # if mask is not empty, if valid k-value exist in the mask
   ...    k = viewer.get_k() #average value of the size of the wave vector
   ...    x, y = viewer.get_data() #averaged data

You have to do this index by index. Another, better way is to work with the normalized data and use k_select iterator, like:

.. doctest:: 

   >>> from cddm.map import k_select
   >>> fast, slow = normalize_multi(data, bg, var, scale = True)
   >>> x,y = log_merge(fast, slow)
   >>> k_data = k_select(y, angle = 0, sector = 30)

Here, k_data is an iterator of (k_avg, data_avg) elements, where k_avg is the mean size of the wavevector and avg_data is the averaged data. You can save this to disk::

   >>> for (k_avg, data_avg) in k_data:
   ...    np.savetxt("data_{}.txt".format(k_avg), data_avg)
  

In the example above we were simulating Brownian motion of particles, so the correlation function decays exponentially, the fitted results are proportional to the square of the wave vector, from where we can obtain the diffusion constant and compare the results with the theoretical prediction. See the source of the plots below for example fitting script if you wish to perform k-averaging and fitting in python.

.. plot:: examples/cross_correlate_k_fit.py

   Here we plot fitted results from the cross-correlation function computed with :func:`.multitau.iccorr_multi`  using subtract_background = False option. For this example, the *norm = 3* datapoint are closest to the theoretically predicted value shown in graph.

.. _imageio: https://github.com/imageio/imageio
.. _paper: https://doi.org/10.1039/C9SM00121B
