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


.. function:: median_decrease(array, minv, maxv, out)

   Performs median decrease filter. Each next element must be smaller or equal


.. function:: median_increase(array, minv, maxv, out)

   Performs median increase filter. Each next element must be greater or equal


