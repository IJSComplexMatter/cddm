"""Tools for cross-differential dynamic microscopy. Core functionality is defined 
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

"""

from __future__ import absolute_import

__version__ = "0.3.0"


