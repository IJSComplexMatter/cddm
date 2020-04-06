:mod:`cddm.video`
=================

.. py:module:: cddm.video

.. autoapi-nested-parse::

   Created on Mon Jul 29 22:01:49 2019

   @author: andrej



Module Contents
---------------

.. function:: fromarrays(arrays)

   Creates a multi-frame iterator from given list of arrays.

   :param arrays: A tuple of array-like objects that represent a single-camera videos
   :type arrays: tuple of array-like

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: asarrays(video, count=None)

   Loads multi-frame video into numpy arrays.

   :param video: A multi-frame iterator object.
   :type video: iterable
   :param count: Defines how many frames are in the video. If not provided and video has
                 an undefined length, it will try to load the video using np.asarray.
                 This means that data copying
   :type count: int, optional


.. function:: asmemmaps(basename, video, count=None)

   Loads multi-frame video into numpy memmaps.

   Actual data is written to numpy files with the provide basename and
   subscripted by source identifier (index), e.g. "basename_0.npy" and "basename_1.npy"
   in case of dual-frame video source.

   :param basename: Base name for the filenames of the videos.
   :type basename: str
   :param video: A multi-frame iterator object.
   :type video: iterable
   :param count: Defines how many multi-frames are in the video. If not provided it is determined
                 by len().
   :type count: int, optional


