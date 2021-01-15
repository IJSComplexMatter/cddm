:mod:`cddm.avg`
===============

.. py:module:: cddm.avg

.. autoapi-nested-parse::

   Data averaging tools.



Module Contents
---------------







.. function:: denoise(x, n=3)

   Denoises data. A sequence of median and convolve filters.

   :param x: Input array
   :type x: ndarray
   :param n: Number of denoise steps (3 by default)
   :type n: int
   :param out: Output array
   :type out: ndarray, optional

   :returns: **out** -- Denoised data.
   :rtype: ndarray


