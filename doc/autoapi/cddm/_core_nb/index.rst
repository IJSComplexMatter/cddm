:orphan:

:mod:`cddm._core_nb`
====================

.. py:module:: cddm._core_nb

.. autoapi-nested-parse::

   Low level numba functions



Module Contents
---------------

.. function:: abs2(x)

   Absolute square of data


.. function:: mean(a, b)

   Man value


.. function:: choose(a, b)

   Chooses data randomly


.. function:: convolve(a, out)

   Convolves input array with kernel [0.25,0.5,0.25]


.. function:: interpolate(x_new, x, y, out)

   Linear interpolation


.. function:: log_interpolate(x_new, x, y, out=None)

   Linear interpolation in semilogx space.


.. function:: median(array, out)

   Performs median filter.


.. function:: weighted_sum(x, y, weight)

   Performs weighted sum of two data sets, given the weight data.
   Weight must be normalized between 0 and 1. Performs:
   `x * weight + (1.- weight) * y`


.. function:: decreasing(array, out)

   Performs decreasing filter. Each next element must be smaller or equal


.. function:: increasing(array, out)

   Performs increasing filter. Each next element must be greater or equal


.. function:: weight_from_g1(g1)

   Computes weight for weighted normalization from normalized and scaled
   correlation function


.. function:: weight_from_d(d)

   Computes weight for weighted normalization from normalized and scaled
   image structure function


