:mod:`cddm.viewer`
==================

.. py:module:: cddm.viewer

.. autoapi-nested-parse::

   A simple matlotlib-based video and correlation data viewers



Module Contents
---------------

.. py:class:: VideoViewer(video, n=None, id=0, title='')

   Bases: :class:`object`

   A matplotlib-based video viewer.

   :param video: A list of a tuple of 2D arrays or a generator of a tuple of 2D arrays.
                 If an iterator is provided, you must set 'n' as well.
   :type video: list-like, iterator
   :param n: Length of the video. When this is set it displays only first 'n' frames of the video.
   :type n: int
   :param id: For multi-frame data specifies camera index.
   :type id: int, optional
   :param title: Plot title.
   :type title: str, optional

   .. method:: show(self)


      Shows video.



.. py:class:: DataViewer(norm=None, scale=False, semilogx=True, shape=None, size=None)

   Bases: :class:`object`

   Shows correlation data in plot. You need to hold reference to this object, otherwise it will not work in interactive mode.

   .. method:: set_data(self, data, background=None, variance=None)


      Sets correlation data.

      :param data: A data tuple (as computed by ccorr, cdiff, adiff, acorr functions)
      :type data: tuple
      :param background: Background data for normalization. For adiff, acorr functions this
                         is ndarray, for cdiff,ccorr, it is a tuple of ndarrays.
      :type background: tuple or ndarray
      :param variance: Variance data for normalization. For adiff, acorr functions this
                       is ndarray, for cdiff,ccorr, it is a tuple of ndarrays.
      :type variance: tuple or ndarray


   .. method:: get_data(self)


      Returns computed k-averaged data and time

      :returns: **x, y** -- Time, data tuple of ndarrays.
      :rtype: ndarray, ndarray


   .. method:: set_mask(self, k, phi=0, sector=5, kstep=1)


      Sets k-mask for averaging,

      :param k: k index in kstep units.
      :type k: int
      :param phi: Mean k-angle in degrees. Measure with respecto to image horizontal axis.
      :type phi: int
      :param sector: Averaging full angle in degrees.
      :type sector: int
      :param kstep: K step in units of minimum k step for a given FFT dimensions.
      :type kstep: float, optional

      :returns: **ok** -- True if mask is valid else False
      :rtype: bool


   .. method:: plot(self)


      Plots data. You must first call :meth:`.set_data` to set input data


   .. method:: show(self)


      Shows plot.



.. py:class:: MultitauViewer(norm=None, scale=False, semilogx=True, shape=None)

   Bases: :class:`cddm.viewer.DataViewer`

   Shows multitau data in plot. You need to hold reference to this object,
   otherwise it will not work in interactive mode.


