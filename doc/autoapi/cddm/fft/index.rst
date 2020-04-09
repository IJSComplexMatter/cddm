:mod:`cddm.fft`
===============

.. py:module:: cddm.fft

.. autoapi-nested-parse::

   FFT tools.

   This module defines several functions for fft processing  of multi-frame data.



Module Contents
---------------

.. function:: show_fft(video, id=0, clip=None, title=None)

   Show fft

   :param video: A multi-frame iterator
   :type video: iterator
   :param id: Frame index
   :type id: int
   :param clip: Clipping value. If not given, it is determined automatically.
   :type clip: float, optional
   :param title: Unique title of the video. You can use :func:`.video.figure_title`
                 to create a unique name.
   :type title: str, optional

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: show_fftdiff(video, clip=None, title=None)

   Show fft difference video

   :param video: A multi-frame iterator
   :type video: iterator
   :param clip: Clipping value. If not given, it is determined automatically.
   :type clip: float, optional
   :param title: Unique title of the video. You can use :func:`figure_title``
                 a to produce unique name.
   :type title: str, optional

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: rfft2(video, kimax=None, kjmax=None, overwrite_x=False)

   A generator that performs rfft2 on a sequence of multi-frame data.

   Shape of the output depends on kimax and kjmax. It is (2*kimax+1, kjmax +1),
   or same as the result of rfft2 if kimax and kjmax are not defined.

   :param video: An iterable of multi-frame data
   :type video: iterable
   :param kimax: Max value of the wavenumber in vertical axis (i)
   :type kimax: float, optional
   :param kjmax: Max value of the wavenumber in horizontal axis (j)
   :type kjmax: float, optional
   :param overwrite_x: If input type is complex and fft library used is not numpy, fft can
                       be performed inplace to speed up computation.
   :type overwrite_x: bool, optional

   :returns: **video** -- An iterator over FFT of the video.
   :rtype: iterator


.. function:: normalize_fft(video, inplace=False, dtype=None)

   Normalizes each frame in fft video to the mean value (intensity) of
   the [0,0] component of fft.

   :param video: Input multi-frame iterable object. Each element of the iterable is a tuple
                 of ndarrays (frames)
   :type video: iterable
   :param inplace: Whether tranformation is performed inplace or not.
   :type inplace: bool, optional
   :param dtype: If specifed, determines output dtype. Only valid if inplace == False.
   :type dtype: numpy dtype

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


