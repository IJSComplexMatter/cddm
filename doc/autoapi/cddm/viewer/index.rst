:mod:`cddm.viewer`
==================

.. py:module:: cddm.viewer

.. autoapi-nested-parse::

   A simple matlotlib-based video and correlation data viewers



Module Contents
---------------

.. py:class:: VideoViewer(video, count=None, id=0, title='', **kw)

   Bases: :class:`object`

   A matplotlib-based video viewer.

   :param video: A list of a tuple of 2D arrays or a generator of a tuple of 2D arrays.
                 If an iterator is provided, you must set 'count' as well.
   :type video: list-like, iterator
   :param count: Length of the video. When this is set it displays only first 'count' frames of the video.
   :type count: int
   :param id: For multi-frame data specifies camera index.
   :type id: int, optional
   :param title: Plot title.
   :type title: str, optional
   :param kw: Extra arguments passed directly to imshow function
   :type kw: options, optional

   .. method:: show(self)


      Shows video.



.. py:class:: DataViewer(semilogx=True, shape=None, mask=None)

   Bases: :class:`object`

   Plots normalized correlation data. You need to hold reference to this object,
   otherwise it will not work in interactive mode.

   :param semilogx: Whether plot data with semilogx or not.
   :type semilogx: bool
   :param shape: Original frame shape. For non-rectangular you must provide this so
                 to define k step.
   :type shape: tuple of ints, optional
   :param size: If specified, perform log_averaging of data with provided size parameter.
                If not given, no averaging is performed.
   :type size: int, optional
   :param mask: A boolean array indicating which data elements were computed.
   :type mask: ndarray, optional

   .. method:: set_data(self, data, t=None)


      Sets correlation data.

      :param data: Normalized data.
      :type data: ndarray


   .. method:: get_data(self)


      Returns computed k-averaged data and time

      :returns: **x, y** -- Time, data ndarrays.
      :rtype: ndarray, ndarray


   .. method:: get_k(self)


      Returns average k value of current data.


   .. method:: set_mask(self, k, angle=0, sector=5, kstep=1)


      Sets k-mask for averaging,

      :param k: k index in kstep units.
      :type k: int
      :param angle: Mean k-angle in degrees. Measure with respecto to image horizontal axis.
      :type angle: int
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



.. py:class:: CorrViewer(semilogx=True, shape=None, size=None, norm=None, scale=False, mask=None)

   Bases: :class:`cddm.viewer.DataViewer`

   Plots raw correlation data. You need to hold reference to this object,
   otherwise it will not work in interactive mode.

   :param semilogx: Whether plot data with semilogx or not.
   :type semilogx: bool
   :param shape: Original frame shape. For non-rectangular you must provide this so
                 to define k step.
   :type shape: tuple of ints, optional
   :param size: If specified, perform log_averaging of data with provided size parameter.
                If not given, no averaging is performed.
   :type size: int, optional
   :param norm: Normalization constant used in normalization
   :type norm: int, optional
   :param scale: Scale constant used in normalization.
   :type scale: bool, optional
   :param mask: A boolean array indicating which data elements were computed.
   :type mask: ndarray, optional

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



.. py:class:: MultitauViewer(semilogx=True, shape=None, norm=None, scale=False, mask=None)

   Bases: :class:`cddm.viewer.CorrViewer`

   Shows multitau data in plot. You need to hold reference to this object,
   otherwise it will not work in interactive mode.

   :param semilogx: Whether plot data with semilogx or not.
   :type semilogx: bool
   :param shape: Original frame shape. For non-rectangular you must provide this so
                 to define k step.
   :type shape: tuple of ints, optional
   :param norm: Normalization constant used in normalization
   :type norm: int, optional
   :param scale: Scale constant used in normalization.
   :type scale: bool, optional
   :param mask: A boolean array indicating which data elements were computed.
   :type mask: ndarray, optional


