:mod:`cddm`
===========

.. py:module:: cddm

.. autoapi-nested-parse::

   Tools for cross-differential dynamic microscopy. Core functionality is defined
   in the following submodules:

   * :mod:`.conf` : Configuration functions and constants.
   * :mod:`.core`: Computation of in-memmory data using standard (linear) algorithms,
     including cross-correlation, cross-difference, for regular- or irregular-spaced
     data and normalization functions. Also linear algorithms for out-of-memmory data.
   * :mod:`.fft` : FFT processing tools.
   * :mod:`.map` : Data mapping and k-averaging functions.
   * :mod:`.multitau`: Computation of in-memory and out-of-memmory data using
     nonlinear (multiple tau) algorithm for regular- and irregular-spaced data and
     normalization functions. Also functions to convert linear data to log-spaced data.
   * :mod:`.video` : Video processing tools.
   * :mod:`.viewer` : Matplotlib-based visualizers of videos and computed data.
   * :mod:`.window` : FFT windowing function.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   avg/index.rst
   conf/index.rst
   core/index.rst
   decorators/index.rst
   fft/index.rst
   map/index.rst
   multitau/index.rst
   norm/index.rst
   print_tools/index.rst
   sim/index.rst
   video/index.rst
   viewer/index.rst
   window/index.rst


