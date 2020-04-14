:mod:`cddm.window`
==================

.. py:module:: cddm.window

.. autoapi-nested-parse::

   Windowing functions.

   These windowing functions can be used to generate FFT window functions.



Module Contents
---------------

.. function:: blackman(shape, out=None)

   Returns a blacman window of a given shape.

   :param shape: A shape of the 2D window.
   :type shape: (int,int)
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- A Blackman window
   :rtype: ndarray


.. function:: gaussian(shape, sigma, out=None)

   Gaussian  window function.

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param sigma: Waist of the gaussian in units of total width of the frame.
   :type sigma: float
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- Gaussian beam window
   :rtype: ndarray


.. function:: tukey(shape, alpha, out=None)

   Returns a tukey window function.

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param alpha: Smoothnes parameter - defines smoothness of the edge of the tukey
                 (should be between 0. and 1. When set to zero, it
                 becomes an aperture filter. When set to 1 it becomes a Hann window.
   :type alpha: float
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- Tukey window
   :rtype: ndarray


.. function:: hann(shape, out=None)

   Returns a hann window function.

   :param shape: A shape of the 2D window
   :type shape: (int,int)
   :param out: Output array.
   :type out: ndarray, optional

   :returns: **window** -- Hann window
   :rtype: ndarray


.. function:: plot_windows(shape=(256, 256), alpha=0.5, sigma=0.4)

   Plots all windows with a given shape, alpha (for tukey), sigma (for gaussian)
   values


