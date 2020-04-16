:mod:`cddm.video`
=================

.. py:module:: cddm.video

.. autoapi-nested-parse::

   Video processing tools.

   You can use this helper functions to perform normalization, background subtraction
   and windowing on multi-frame data.

   There are also function for real-time display of videos for real-time analysis.



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
   :param count: Defines how many frames are in the video. If not provided it will calculate
                 length of the video based on the length of the iterable. If that is not
                 possible ValueError is raised
   :type count: int, optional

   :returns: **out** -- A tuple of array(s) representing video(s)
   :rtype: tuple of arrays


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

   :returns: **out** -- A tuple of memmapped array(s) representing video(s)
   :rtype: tuple of arrays


.. function:: load(video, count=None)

   Loads video into memory.

   :param video: A multi-frame iterator object.
   :type video: iterable
   :param count: Defines how many frames are in the video. If not provided it will calculate
                 length of the video based on the length of the iterable. If that is not
                 possible ValueError is raised
   :type count: int, optional

   :returns: **out** -- A video iterable. A tuple of multi-frame data (arrays)
   :rtype: tuple


.. function:: crop(video, roi=(slice(None), slice(None)))

   Crops each frame in the video.

   :param video: Input multi-frame iterable object. Each element of the iterable is a tuple
                 of ndarrays (frames)
   :type video: iterable
   :param roi: A tuple of two slice objects for slicing in first axis (height) and the
               second axis (width). You can also provide a tuple arguments tuple
               that are past to the slice builtin function.
   :type roi: tuple

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator

   .. rubric:: Examples

   One option is to provide roi with indices. To crop frames like frame[10:100,20:120]

   >>> video = random_video(count = 100)
   >>> video = crop(video, roi = ((10,100),(20,120)))

   Or you can use slice objects to perform crop

   >>> video = crop(video, roi = (slice(10,100),slice(20,120)))


.. function:: subtract(x, y, inplace=False, dtype=None)

   Subtracts two videos.

   :param x, y: Input multi-frame iterable object. Each element of the iterable is a tuple
                of ndarrays (frames)
   :type x, y: iterable
   :param inplace: Whether tranformation is performed inplace or not.
   :type inplace: bool, optional
   :param dtype: If specifed, determines output dtype. Only valid if inplace == False.
   :type dtype: numpy dtype

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: add(x, y, inplace=False, dtype=None)

   Adds two videos.

   :param x, y: Input multi-frame iterable object. Each element of the iterable is a tuple
                of ndarrays (frames)
   :type x, y: iterable
   :param inplace: Whether tranformation is performed inplace or not.
   :type inplace: bool, optional
   :param dtype: If specifed, determines output dtype. Only valid if inplace == False.
   :type dtype: numpy dtype

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: normalize_video(video, inplace=False, dtype=None)

   Normalizes each frame in the video to the mean value (intensity).

   :param video: Input multi-frame iterable object. Each element of the iterable is a tuple
                 of ndarrays (frames)
   :type video: iterable
   :param inplace: Whether tranformation is performed inplace or not.
   :type inplace: bool, optional
   :param dtype: If specifed, determines output dtype. Only valid if inplace == False.
   :type dtype: numpy dtype

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: multiply(x, y, inplace=False, dtype=None)

   Multiplies two videos.

   :param x,y: Input multi-frame iterable object. Each element of the iterable is a tuple
               of ndarrays (frames)
   :type x,y: iterable
   :param inplace: Whether tranformation is performed inplace or not.
   :type inplace: bool, optional
   :param dtype: If specifed, determines output dtype. Only valid if inplace == False.
   :type dtype: numpy dtype

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. py:class:: ImageShow(title='video', norm_func=lambda x: x)

   A simple interface for video visualization using matplotlib opencv or
   pyqtgraph.

   :param title: Title of the video
   :type title: str
   :param norm_func: Normalization function that takes a single argument (array) and returns
                     a single element (array). Can be used to apply custom normalization
                     function to the image before it is shown.
   :type norm_func: callable

   .. method:: show(self, im)


      Shows image

      :param im: A 2D array
      :type im: ndarray



.. function:: pause(i=1)

   Pause in milliseconds needed to update matplotlib or opencv figures


.. function:: play(video, fps=100.0, max_delay=0.1)

   Plays video for real-time visualization.

   You must first call show functions (e.g. :func:`show_video`) to specify
   what needs to be played. This function performs the actual display when in
   a for loop

   :param video: A multi-frame iterable object.
   :type video: iterable
   :param fps: Expected FPS of the input video. If rendering of video is too slow
               for the expected frame rate, frames will be skipped to assure the
               expected acquisition. Therefore, you must match exactly the acquisition
               frame rate with this parameter.
   :type fps: float
   :param max_delay: Max delay that visualization can produce before it starts skipping frames.
   :type max_delay: float

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator

   .. rubric:: Examples

   First create some test data of a dual video

   >>> video = random_video(count = 256, dual = True)
   >>> video = show_video(video)

   Now we can load video to memory, and play it as we load frame by frame...

   >>> v1,v2 = asarrays(play(video, fps = 30),count = 256)


.. function:: figure_title(name)

   Generate a unique figure title


.. function:: norm_rfft2(clip=None, mode='real')

   Returns a frame normalizing function for :func:`show_video`


.. function:: show_fft(video, id=0, title=None, clip=None, mode='real')

   Show fft of the video.

   :param video: A multi-frame iterator
   :type video: iterator
   :param id: Frame index
   :type id: int
   :param title: Unique title of the video. You can use :func:`.video.figure_title`
                 to create a unique name.
   :type title: str, optional
   :param clip: Clipping value. If not given, it is determined automatically.
   :type clip: float, optional
   :param mode: What to display, "real", "imag" or "abs"
   :type mode: str

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: show_video(video, id=0, title=None, norm_func=lambda x: x.real)

   Returns a video and performs image live video show.
   This works in connection with :func:`play` that does the actual display.

   :param video: A multi-frame iterator
   :type video: iterator
   :param id: Frame index
   :type id: int
   :param title: Unique title of the video. You can use :func:`figure_title`
                 a to produce unique name.
   :type title: str
   :param norm_func: Normalization function that takes a single argument (array) and returns
                     a single element (array). Can be used to apply custom normalization
                     function to the image before it is shown.
   :type norm_func: callable

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: show_diff(video, title=None)

   Returns a video and performs image difference live video show.
   This works in connection with :func:`play` that does the actual display.

   :param video: A multi-frame iterator
   :type video: iterator
   :param title: Unique title of the video. You can use :func:`figure_title`
                 a to produce unique name.
   :type title: str

   :returns: **video** -- A multi-frame iterator
   :rtype: iterator


.. function:: random_video(shape=(512, 512), count=256, dtype=FDTYPE, max_value=1.0, dual=False)

   Random multi-frame video generator, useful for testing.


