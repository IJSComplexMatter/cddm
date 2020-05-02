:mod:`cddm.avg`
===============

.. py:module:: cddm.avg

.. autoapi-nested-parse::

   Data averaging tools.



Module Contents
---------------

.. function:: log_interpolate(x_new, x, y, out)

   Linear interpolation in semilogx space.


.. function:: decreasing(array, out)

   Performs decreasing filter. Each next element must be smaller or equal


.. function:: increasing(array, out)

   Performs increasing filter. Each next element must be greater or equal


.. function:: median(array, out)

   Performs median filter.


.. function:: convolve(a, out)

   Convolves input array with kernel [0.25,0.5,0.25]


.. function:: interpolate(x_new, x, y, out)

   Linear interpolation


.. function:: weighted_sum(x, y, weight)

   Performs weighted sum of two data sets, given the weight data.
   Weight must be normalized between 0 and 1. Performs:
   `x * weight + (1.- weight) * y`


.. function:: denoise(x, n=3, out=None)

   Denoises data. A sequence of median and convolve filters.

   :param x: Input array
   :type x: ndarray
   :param n: Number of denoise steps (3 by default)
   :type n: int
   :param out: Output array
   :type out: ndarray, optional

   :returns: **out** -- Denoised data.
   :rtype: ndarray


.. function:: base_weight(corr, scale_factor=1.0, mode='corr', pre_filter=True, out=None)

   Computes weighting function for baseline weighted normalization.

   :param corr: Correlation (or difference) data
   :type corr: ndarray
   :param scale_factor: Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
                        corr data must be computed with scale = True option.
   :type scale_factor: ndarray
   :param mode: Representation mode, either 'corr' (default) or 'diff'
   :type mode: str
   :param pre_filter: Whether to perform denoising and filtering. If set to False, user has
                      to perform data filtering.
   :type pre_filter: bool
   :param out: Output array
   :type out: ndarray, optional

   :returns: **out** -- Weight data for weighted sum calculation.
   :rtype: ndarray


.. function:: comp_weight(corr, scale_factor=1.0, mode='corr', pre_filter=True, out=None)

   Computes weighting function for compensating weighted normalization.

   :param corr: Correlation (or difference) data
   :type corr: ndarray
   :param scale_factor: Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
                        corr data must be computed with scale = True option.
   :type scale_factor: ndarray
   :param mode: Representation mode, either 'corr' (default) or 'diff'
   :type mode: str
   :param pre_filter: Whether to perform denoising and filtering. If set to False, user has
                      to perform data filtering.
   :type pre_filter: bool
   :param out: Output array
   :type out: ndarray, optional

   :returns: **out** -- Weight data for weighted sum calculation.
   :rtype: ndarray


.. function:: weight_from_data(x, xp, yp, scale_factor=1.0, mode='corr', norm=WEIGHT_BASELINE, pre_filter=True)

   Computes weight at given x values from correlation data points (xp, yp).

   :param x: x-values of the interpolated values
   :type x: ndarray
   :param xp: x-values of the correlation data
   :type xp: ndarray
   :param yp: y-values of the correlation data, correlation data is over the last axis.
   :type yp: ndarray
   :param scale_factor: Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
                        corr data must be computed with scale = True option.
   :type scale_factor: ndarray
   :param mode: Representation mode, either 'corr' (default) or 'diff'
   :type mode: str
   :param norm: Weighting mode, 1 or odd for compensated, 0 or even for baseline.
   :type norm: int
   :param pre_filter: Whether to perform denoising and filtering. If set to False, user has
                      to perform data filtering.
   :type pre_filter: bool

   :returns: **out** -- Weight data for weighted sum calculation.
   :rtype: ndarray


