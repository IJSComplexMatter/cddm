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

   Chooses data, randomly


.. function:: convolve(a, out)

   Convolves input array with kernel [0.25,0.5,0.25]


.. function:: median(array, out)

   Performs median filter.


.. function:: decreasing(array, out)

   Performs decreasing filter. Each next element must be smaller or equal


.. function:: increasing(array, out)

   Performs increasing filter. Each next element must be greater or equal


