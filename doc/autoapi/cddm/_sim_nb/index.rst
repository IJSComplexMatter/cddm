:orphan:

:mod:`cddm._sim_nb`
===================

.. py:module:: cddm._sim_nb

.. autoapi-nested-parse::

   Numba functions for sim module



Module Contents
---------------

.. function:: mirror(x, x0, x1)

   transforms coordinate x by flooring in the interval of [x0,x1]
   It performs x0 + (x-x0)%(x1-x0)


.. function:: numba_seed(value)

   Seed for numba random generator


.. function:: make_step(x, scale, velocity)

   Performs random particle step from a given initial position x.


.. function:: psf_gauss(x, x0, y, y0, sigma, intensity)

   Gaussian point-spread function. This is used to calculate pixel value
   for a given pixel coordinate x,y and particle position x0,y0.


.. function:: draw_points(im, points, intensity)

   Draws pixels to image from a given points array


.. function:: draw_psf(im, points, intensity, sigma)

   Draws psf to image from a given points array


