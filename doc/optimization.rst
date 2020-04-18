.. _optimization:

Configuration & Tips
====================

In the :mod:`cddm.conf` there are a few configuration options that you can use for custom configuration, optimization and tuning. The package relies heavily on numba-optimized code. Default numba compilation options are used. For fine-tuning you can of course use custom compilation options that numba provides (See numba_ compilation options). There are also a few numba related environment variables that you can set in addition to numba_ compilation options. These are explained below.

Runtime options
---------------

Verbosity
+++++++++

By default, compute functions do not print to stdout. You can set printing of progress bar and messages with:

.. doctest::

   >>> import cddm.conf
   >>> cddm.conf.set_verbose(1) #level 1 messages
   0
   >>> cddm.conf.set_verbose(2) #level 2 messages (more info)
   1

To disable verbosity, set verbose level to zero:

.. doctest::

   >>> cddm.conf.set_verbose(0) #disable printing to stdout
   2

.. note:: 

   The setter functions in the :mod:`cddm.conf` module return previous defined setting.

You can set this option in the configuration file (see below).

Video preview
+++++++++++++

Video previewing can be done using `cv2` or `pyqtgraph` (in installed) instead of `matplotlib`::

   >>> cddm.conf.set_showlib("cv2")
   "matplotlib"

You can set this option in the configuration file (see below).

FFT library
+++++++++++

You can select FFT library ("mkl_fft", "numpy", "scipy" or "pyfftw") for rfft2 calculation, and for fff calculations triggered with method = "fft" in the correlation functions::

   >>> cddm.conf.set_fftlib("mkl_fft")
   'mkl_fft'
   >>> cddm.conf.set_rfft2lib("numpy")
   'numpy'

Compilation options
-------------------

Numba multithreading
++++++++++++++++++++

Most computationally expensive numerical algorithms were implemented using @vectorize or @guvecgorize and can be compiled with target="parallel" option. By default, parallel execution is disabled.

You can enable parallel target for numba functions by setting the *CDDM_TARGET_PARALLEL* environment variable. This has to be set prior to importing the package.

.. doctest::

   >>> import os
   >>> os.environ["CDDM_TARGET_PARALLEL"] = "1"
   >>> import cddm #parallel enabled cddm

Another option is to modify the configuration file (see below). Depending on the number of cores in your system, you should be able to notice an increase  in the computation speed.

Numba cache
+++++++++++

Numba allows caching of compiled functions. If *CDDM_TARGET_PARALLEL* environment variable is not defined, all compiled functions are cached and stored in your home directory for faster import by default. For debugging purposes, you can enable/disable caching with *CDDM_NUMBA_CACHE* environment variable. To disable caching (enabled by default):

.. doctest::

   >>> os.environ["CDDM_NUMBA_CACHE"]  = "0"

Cached files are stored in *.cddm/numba_cache*  in user's home directory. You can remove this folder to force recompilation. To enable/disable caching you can modify the configuration file (see below).

Precision
+++++++++

By default, computation is performed in double precision. You may disable double precision if you are low on memory, and to gain some speed in computation. 

.. doctest::

   >>> os.environ["CDDM_DOUBLE_PRECISION"] = "0"

You can also use *fastmath* option in numba compilation to gain some small speed by reducing the computation accuracy when using MKL.

   >>> os.environ["CDDM_FASTMATH"] = "1"

Default values can also be set the configuration file (see below).

Optimization tips
-----------------

Is the computation too slow? 

Here you can find some tips for optimizing your code to speed up the calculation, should you need this. I suggest you work with some test data that you read into memory, and then use::

   >>> cddm.conf.set_cerbose(2) 
   0

so that it will plot the execution speed. First make sure you are running in multiple threads (see compilation options). Then as you work on your dataset, test how the following options change the computation speed:

* You can select the method = 'diff' method and norm = 1 to force calculation without the extra `data_sum` arrays. Or, calculate with norm = 0 and method = "corr".
* For non-iterative version, you can also use `align` = True and try to see if copying and aligning the data in memory before the calculation improves.
* Use k masks to compute only the part of k-space that interests you.

Multiple tau algorithm
++++++++++++++++++++++

The multiple tau algorithm is the best option, if you want log-spaced results. Here are some multiple-tau-specific options that you can tune

* You can speed up the computation if you lower the `level_size` parameter, which effectively reduces the time resolution.
* Play with `chunk_size` parameter and find the best option for your dataset.
* Use `thread_divisor` and find the best combination of `chunk_size` and `thread_divisor` (see below).

.. note::

   Internally, the algorithm uses numba `vectorize` and `guvectorize` for automatic multi-threading. However, the efficiency of the threaded calculation depends on the input data shape and cache size of your processor. With `thread_divisor` and `chunk_size` parameters  you are reshaping the input data, which might help in speeding up. 

Linear algorithm
++++++++++++++++

If you need linear data, the fft algorithm works best for regular-spaced data and complete tau calculation. Here you may work with floats instead of doubles to speed up FFT or work with different fft library. See Optimization for details. If you can work with a limited range of delay times you may use the `n` parameter, which may be speedier to compute with the standard `method = "corr"` Here, you should use `align = True`. You can also use the :func:`.core.reshape_input` and  :func:`.core.reshape_output` with combination of `thread_divisor` parameter.

Thread divisor
++++++++++++++

If you are doing data masking, the input video frames are shaped to 1D. Consequently, computation runs on a single thread. You can reshape the data from 1D to 2D, which will trigger multithreaded calculation. The length of the first axis of the data frame determines the number of thread jobs that will run. Note that you can reshape also 2D data to optimize array size that each thread is working on, which might improve the speed of execution. The out-of-memory functions and multitau algorithm allows you to reshape the data on-the-fly with the `thread_divisor`. The in-memory calculation of the standard (linear) correlation function does not support `thread_divisor` options. Instead, you can do::

   >>> from cddm.core import reshape_input, reshape_output
   >>> fft_reshaped, original_shape = reshape_input(fft_array, thread_divisor = 8)
   >>> acorr = acorr(fft_reshaped)
   >>> acorr = reshape_output(acorr, original_shape) #reshape back

.. note::

   When `thread_divisor` is defined, input data is reshaped from any shape (x,y...) to (thread_divisor, rest). So thread_divisor must be a divisor of the data size, otherwise error is raised. Therefore, you must use this parameter in combination with `kimax` and `kjmax`, that define your data size. If you use mask arrays, the thread_divisor must divide the length of masked data size. 


CDDM configuration file
-----------------------

You can also edit the configuration file *.cddm/cddm.ini* in user's home directory to define default settings. This file is automatically generated from a template if it does not exist in the directory.

.. literalinclude:: cddm.ini


.. _numba: https://numba.pydata.org/numba-doc/latest/reference/envvars.html

