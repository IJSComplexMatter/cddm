:mod:`cddm.multitau`
====================

.. py:module:: cddm.multitau

.. autoapi-nested-parse::

   Multiple-tau algorithm and log averaging functions.

   Multitau analysis (in-memory, general data versions):

   * :func:`ccorr_multi` to calculate cross-correlation/difference functions.
   * :func:`acorr_multi` to calculate auto-correlation/difference functions.

   For out-of-memory analysis and real-time calculation use:

   * :func:`iccorr_multi` to calculate cross-correlation/difference functions
   * :func:`iacorr_multi` to calculate auto-correlation/difference functions

   For normalization and merging of the results of the above functions use:

   * :func:`normalize_multi` to normalize the outputs of the above functions.
   * :func:`log_merge` To merge results of the multi functions into one continuous data.

   For time averaging of linear spaced data use:

   * :func:`multilevel` to perform multi level averaging.
   * :func:`merge_multilevel` to merge multilevel in a continuous log spaced data
   * :func:`log_average` to perform both of these in one step.



Module Contents
---------------

.. data:: BINNING_SLICE
   :annotation: = 0

   No binning (select every second element)


.. data:: BINNING_MEAN
   :annotation: = 1

   Binning (take mean value)


.. data:: BINNING_CHOOSE
   :annotation: = 2

   Binning with random selection


.. function:: mean(a, b)

   Man value


.. function:: mean_data(data, axis=0, out_axis=None)

   Binning function. Takes data and performs channel binning over a specifed
   axis. If specified, also moves axis to out_axis.


.. function:: choose(a, b)

   Chooses data, randomly


.. function:: choose_data(data, axis=0, out_axis=None)

   Instead of binning, this randomly selects data.
   If specified, also moves axis to out_axis.


.. function:: slice_data(data, axis=0, out_axis=None)

   Slices data so that it takes every second channel. If specified, also moves axis to out_axis.


.. function:: ccorr_multi(f1, f2, t1=None, t2=None, level_size=2**4, norm=None, method=None, align=False, axis=0, period=1, binning=None, nlevel=None, thread_divisor=None, mask=None)

   Multitau version of :func:`.core.ccorr`

   :param f1: A complex ND array of the first data.
   :type f1: array-like
   :param f2: A complex ND array of the second data.
   :type f2: array-like
   :param t1: Array of integers defining frame times of the first data. If not provided,
              regular time-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Array of integers defining frame times of the second data. If not provided,
              regular time-spaced data is assumed.
   :type t2: array-like, optional
   :param level_size: If provided, determines the length of the output.
   :type level_size: int, optional
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param align: Whether to align data prior to calculation. Note that a complete copy of
                 the data takes place.
   :type align: bool, optional
   :param axis: Axis over which to calculate.
   :type axis: int, optional
   :param period: Period of the irregular-spaced random triggering sequence. For regular
                  spaced data, this should be set to 1 (deefault).
   :type period: int, optional
   :param binning: Binning mode (0 - no binning, 1 : average, 2 : random select)
   :type binning: int, optional
   :param nlevel: If specified, defines how many levels are used in multitau algorithm.
                  If not provided, all available levels are used.
   :type nlevel: int, optional
   :param thread_divisor: If specified, input frame is reshaped to 2D with first axis of length
                          specified with the argument. It defines how many treads are run. This
                          must be a divisor of the total size of the frame. Using this may speed
                          up computation in some cases because of better memory alignment and
                          cache sizing.
   :type thread_divisor: int, optional

   :returns: **lin, multi** -- A tuple of linear (short delay) data and multilevel (large delay) data
             See :func:`.core.ccorr` for definition of ccorr_type
   :rtype: ccorr_type, ccorr_type


.. function:: acorr_multi(f, t=None, level_size=2**4, norm=None, method=None, align=False, axis=0, period=1, binning=None, nlevel=None, thread_divisor=None, mask=None)

   Multitau version of :func:`.core.acorr`

   :param f: A complex ND array
   :type f: array-like
   :param t: Array of integers defining frame times of the data. If not provided,
             regular time-spaced data is assumed.
   :type t: array-like, optional
   :param level_size: If provided, determines the length of the output.
   :type level_size: int, optional
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param align: Whether to align data prior to calculation. Note that a complete copy of
                 the data takes place.
   :type align: bool, optional
   :param axis: Axis over which to calculate.
   :type axis: int, optional
   :param period: Period of the irregular-spaced random triggering sequence. For regular
                  spaced data, this should be set to 1 (deefault).
   :type period: int, optional
   :param binning: Binning mode (0 - no binning, 1 : average, 2 : random select)
   :type binning: int, optional
   :param nlevel: If specified, defines how many levels are used in multitau algorithm.
                  If not provided, all available levels are used.
   :type nlevel: int, optional
   :param thread_divisor: If specified, input frame is reshaped to 2D with first axis of length
                          specified with the argument. It defines how many treads are run. This
                          must be a divisor of the total size of the frame. Using this may speed
                          up computation in some cases because of better memory alignment and
                          cache sizing.
   :type thread_divisor: int, optional

   :returns: **lin, multi** -- A tuple of linear (short delay) data and multilevel (large delay) data
             See :func:`.core.acorr` for definition of acorr_type
   :rtype: acorr_type, acorr_type


.. function:: iccorr_multi(data, t1=None, t2=None, level_size=2**4, norm=3, method='corr', count=None, period=1, binning=None, nlevel=None, chunk_size=None, thread_divisor=None, auto_background=False, viewer=None, viewer_interval=1, mode='full', mask=None, stats=True)

   Iterative version of :func:`.ccorr_multi`

   :param data: An iterable object, iterating over dual-frame ndarray data.
   :type data: iterable
   :param t1: Array of integers defining frame times of the first data.  If not defined,
              regular-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Array of integers defining frame times of the second data. If t1 is defined,
              you must define t2 as well.
   :type t2: array-like, optional
   :param level_size: If provided, determines the length of the multi_level data.
   :type level_size: int, optional
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param count: If given, it defines how many elements of the data to process. If not given,
                 count is set to len(t1) if that is not specified, it is set to len(data).
   :type count: int, optional
   :param period: Period of the irregular-spaced random triggering sequence. For regular
                  spaced data, this should be set to 1 (deefault).
   :type period: int, optional
   :param binning: Binning mode (0 - no binning, 1 : average, 2 : random select)
   :type binning: int, optional
   :param nlevel: If specified, defines how many levels are used in multitau algorithm.
                  If not provided, all available levels are used.
   :type nlevel: int, optional
   :param chunk_size: Length of data chunk.
   :type chunk_size: int
   :param thread_divisor: If specified, input frame is reshaped to 2D with first axis of length
                          specified with the argument. It defines how many treads are run. This
                          must be a divisor of the total size of the frame. Using this may speed
                          up computation in some cases because of better memory alignment and
                          cache sizing.
   :type thread_divisor: int, optional
   :param auto_background: Whether to use data from first chunk to calculate and subtract background.
   :type auto_background: bool
   :param viewer: You can use :class:`.viewer.MultitauViewer` to display data.
   :type viewer: any, optional
   :param viewer_interval: A positive integer, defines how frequently are plots updated 1 for most
                           frequent, higher numbers for less frequent updates.
   :type viewer_interval: int, optional
   :param mode: Either "full" or "partial". With mode = "full", output of this function
                is identical to the output of :func:`ccorr_multi`. With mode = "partial",
                cross correlation between neighbouring chunks is not computed.
   :type mode: str
   :param mask: If specifed, computation is done only over elements specified by the mask.
                The rest of elements are not computed, np.nan values are written to output
                arrays.
   :type mask: ndarray, optional
   :param stats: Whether to return stats as well.
   :type stats: bool

   :returns: * **(lin, multi), bg, var** (*(ccorr_type, ccorr_type), ndarray, ndarray*) -- A tuple of linear (short delay) data and multilevel (large delay) data,
               background and variance data. See :func:`.core.ccorr` for definition
               of ccorr_type
             * **lin, multi** (*ccorr_type, ccorr_type*) -- If `stats` == False


.. function:: iacorr_multi(data, t=None, level_size=2**4, norm=3, method='corr', count=None, period=1, binning=None, nlevel=None, chunk_size=None, thread_divisor=None, auto_background=False, viewer=None, viewer_interval=1, mode='full', mask=None, stats=True)

   Iterative version of :func:`.acorr_multi`

   :param data: An iterable object, iterating over dual-frame ndarray data.
   :type data: iterable
   :param t: Array of integers defining frame times of the first data. If not defined,
             regular-spaced data is assumed.
   :type t: array-like, optional
   :param level_size: If provided, determines the length of the multi_level data.
   :type level_size: int, optional
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param count: If given, it defines how many elements of the data to process. If not given,
                 count is set to len(t1) if that is not specified, it is set to len(data).
   :type count: int, optional
   :param period: Period of the irregular-spaced random triggering sequence. For regular
                  spaced data, this should be set to 1 (deefault).
   :type period: int, optional
   :param binning: Binning mode (0 - no binning, 1 : average, 2 : random select)
   :type binning: int, optional
   :param nlevel: If specified, defines how many levels are used in multitau algorithm.
                  If not provided, all available levels are used.
   :type nlevel: int, optional
   :param chunk_size: Length of data chunk.
   :type chunk_size: int
   :param thread_divisor: If specified, input frame is reshaped to 2D with first axis of length
                          specified with the argument. It defines how many treads are run. This
                          must be a divisor of the total size of the frame. Using this may speed
                          up computation in some cases because of better memory alignment and
                          cache sizing.
   :type thread_divisor: int, optional
   :param auto_background: Whether to use data from first chunk to calculate and subtract background.
   :type auto_background: bool
   :param viewer: You can use :class:`.viewer.MultitauViewer` to display data.
   :type viewer: any, optional
   :param viewer_interval: A positive integer, defines how frequently are plots updated 1 for most
                           frequent, higher numbers for less frequent updates.
   :type viewer_interval: int, optional
   :param mode: Either "full" or "chunk". With mode = "full", output of this function
                is identical to the output of :func:`ccorr_multi`. With mode = "chunk",
                cross correlation between neighbouring chunks is not computed.
   :type mode: str
   :param mask: If specifed, computation is done only over elements specified by the mask.
                The rest of elements are not computed, np.nan values are written to output
                arrays.
   :type mask: ndarray, optional
   :param stats: Whether to return stats as well.
   :type stats: bool

   :returns: * **(lin, multi), bg, var** (*(acorr_type, acorr_type), ndarray, ndarray*) -- A tuple of linear (short delay) data and multilevel (large delay) data,
               background and variance data. See :func:`.core.acorr` for definition
               of acorr_type
             * **lin, multi** (*acorr_type, acorr_type*) -- If `stats` == False


.. function:: convolve(a, out)

   Convolves input array with kernel [0.25,0.5,0.25]


.. function:: multilevel(data, level_size=16)

   Computes a multi-level version of the linear time-spaced data.

   :param data: Normalized correlation data
   :type data: ndarray
   :param level_size: Level size
   :type level_size: int

   :returns: **x** -- Multilevel data array. Shape of this data depends on the length of the original
             data and the provided parameter
   :rtype: ndarray


.. function:: merge_multilevel(data, mode='full')

   Merges multilevel data (data as returned by the :func:`multilevel` function)

   :param data: data as returned by :func:`multilevel`
   :type data: ndarray
   :param mode: Either 'full' or 'half'. Defines how data from the zero-th level of the
                multi-level data is treated. Take all data (full) or only second half
   :type mode: str, optional

   :returns: **x, y** -- Time, log-spaced data arrays
   :rtype: ndarray, ndarray


.. function:: log_average(data, size=8)

   Performs log average of normalized linear-spaced data.

   You must first normalize with :func:`.core.normalize` before averaging!

   :param data: Input array of linear-spaced data
   :type data: array
   :param size: Sampling size. Number of data points per each doubling of time.
                Any positive number is valid.
   :type size: int

   :returns: **x, y** -- Time and log-spaced data arrays.
   :rtype: ndarray, ndarray


.. function:: log_merge(lin, multi)

   Merges normalized multi-tau data.

   You must first normalize with :func:`normalize_multi` before merging!
   This function performs a multilevel split on the fast (linear) data and
   merges that with the multilevel slow data into a continuous log-spaced
   data.

   :param lin: Linear data
   :type lin: ndarray
   :param multi: Multilevel data
   :type multi: ndarray

   :returns: **x, y** -- Time and log-spaced data arrays.
   :rtype: ndarray, ndarray


.. function:: normalize_multi(*args, **kwargs)

   A multitau version of :func:`.core.normalize`.

   Performs normalization of data returned by :func:`ccorr_multi`,
    :func:`acorr_multi`,:func:`iccorr_multi`, or :func:`iacorr_multi` function.

   See documentation of :func:`.core.normalize`.


