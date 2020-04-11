:mod:`cddm.core`
================

.. py:module:: cddm.core

.. autoapi-nested-parse::

   Core functionality and main computation functions are defined here. There are
   low level implementations.

   * :func:`cross_correlate`
   * :func:`cross_correlate_fft`
   * :func:`auto_correlate`
   * :func:`auto_correlate_fft`
   * :func:`cross_difference`
   * :func:`auto_difference`

   along with functions for calculating tau-dependent mean signal and mean square
   of the signal that are needed for normalization.

   * :func:`cross_sum`
   * :func:`cross_sum_fft`
   * :func:`auto_sum`
   * :func:`auto_sum_fft`

   High-level functions include (in-memory calculation):

   * :func:`ccorr` to calculate cross-correlation/difference functions
   * :func:`acorr` to calculate auto-correlation/difference functions

   For out-of-memory analysis use:

   * :func:`iccorr` to calculate cross-correlation/difference functions
   * :func:`iacorr` to calculate auto-correlation/difference functions

   Finally, normalization of the results:

   * :func:`normalize` to normalize the outputs of the above functions.



Module Contents
---------------

.. data:: NORM_BASELINE
   :annotation: = 0

   baseline normalization


.. data:: NORM_COMPENSATED
   :annotation: = 1

   compensated normalization (cross-diff)


.. data:: NORM_SUBTRACTED
   :annotation: = 2

   background subtraction normalization


.. function:: abs2(x)

   Absolute square of data


.. function:: cross_correlate_fft(f1, f2, t1=None, t2=None, axis=0, n=None, aout=None)

   Calculates cross-correlation function of two equal sized input arrays using FFT.

   For large arrays and large n, this is faster than correlate. The output of
   this function is identical to the output of cross_correlate.

   See :func:`cross_correlate` for details.


.. function:: auto_correlate_fft(f, t=None, axis=0, n=None, aout=None)

   Calculates auto-correlation function of input array using FFT.

   For large arrays and large n, this is faster than correlate. The output of
   this function is identical to the output of auto_correlate.

   See :func:`auto_correlate` for details.


.. function:: thread_frame_shape(shape, thread_divisor=None)

   Computes new frame shape for threaded computaton.

   :param shape: Input frame shape
   :type shape: tuple of ints
   :param thread_divisor: An integer that divides the flattend frame shape. This number determines
                          number of threads.
   :type thread_divisor: int

   :returns: **shape** -- A length 2 shape
   :rtype: tuple


.. function:: reshape_input(f, axis=0, thread_divisor=None, mask=None)

   Reshapes input data, for faster threaded calculation

   :param f: Input array
   :type f: ndarray
   :param axis: Axis over which the computation is performed.
   :type axis: int
   :param thread_divisor: An integer that divides the flattend frame shape. This number determines
                          number of threads.
   :type thread_divisor: int
   :param mask: A boolean mask array. If provided, input data is masked first, then reshaped.
                This only works woth axis = 0.
   :type mask: ndarray

   :returns: **array, old_shape** -- Reshaped array and old frame shape tuple. Old frame shape is needed dor
             reshaping of output data with :func:`reshape_output`
   :rtype: ndarray, tuple


.. function:: reshape_output(data, shape, mask=None)

   Reshapes output data as returned from ccorr,acorr functions
   to original frame shape data.


   If you used :func:`reshape_input` to reshape input data before call to `ccorr`
   or `acorr` functions. You must call this function on the output data to
   reshape it back to original shape and unmasked input array. Missing data
   is filled with np.nan.

   :param data: Data as returned by :func:`acorr` or :func:`ccorr`
   :type data: tuple of ndarrays
   :param shape: shape of the input frame data.
   :type shape: tuple of ints
   :param mask: If provided, reconstruct reshaped data to original shape, prior to masking
                with mask.
   :type mask: ndarray, optional

   :returns: **out** -- Reshaped data, as if there was no prior call to reshape_input on the
             input data of :func:`acorr` or :func:`ccorr` functions.
   :rtype: ndarray


.. function:: cross_correlate(f1, f2, t1=None, t2=None, axis=0, n=None, align=False, aout=None)

   Calculates cross-correlation function of two equal sized input arrays.

   This function performs
   out[k] = sum_{i,j, where  k = abs(t1[i]-t2[j])} (real(f1[i]*conj(f2[j]))

   :param f1: First input array
   :type f1: array-like
   :param f2: Second input array
   :type f2: array-like
   :param t1: First time sequence. If not given, regular-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Second time sequence. If not given, t1 time sequence is assumed.
   :type t2: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Computed cross-correlation.
   :rtype: ndarray

   .. seealso:: :func:`ccorr`, :func:`cddm.multitau.ccorr_multi`


.. function:: cross_difference(f1, f2, t1=None, t2=None, axis=0, n=None, align=False, aout=None)

   Calculates cross-difference (image structure) function of two equal
   sized input arrays.

   This function performs
   out[k] = sum_{i,j, where  k = abs(t1[i]-t2[j])} (abs(f2[j]-f1[i]))**2

   :param f1: First input array
   :type f1: array-like
   :param f2: Second input array
   :type f2: array-like
   :param t1: First time sequence. If not given, regular-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Second time sequence. If not given, t1 time sequence is assumed.
   :type t2: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Computed cross-difference.
   :rtype: ndarray

   .. seealso:: :func:`ccorr`, :func:`cddm.multitau.ccorr_multi`


.. function:: auto_correlate(f, t=None, axis=0, n=None, align=False, aout=None)

   Calculates auto-correlation function.

   This function performs
   out[k] = sum_{i,j, where  k = j - i >= 0} (real(f[i]*conj(f[j]))

   :param f: Input array
   :type f: array-like
   :param t: Time sequence. If not given, regular-spaced data is assumed.
   :type t: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Computed auto-correlation.
   :rtype: ndarray

   .. seealso:: :func:`acorr`, :func:`cddm.multitau.acorr_multi`


.. function:: auto_difference(f, t=None, axis=0, n=None, align=False, aout=None)

   Calculates auto-difference function.

   This function performs
   out[k] = sum_{i,j, where  k = j - i >= 0} np.abs((f[i] - f[j]))**2

   :param f: Input array
   :type f: array-like
   :param t: Time sequence. If not given, regular-spaced data is assumed.
   :type t: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Computed auto-difference.
   :rtype: ndarray

   .. seealso:: :func:`acorr`, :func:`cddm.multitau.acorr_multi`


.. function:: cross_count(t1, t2=None, n=None, aout=None)

   Culculate number of occurences of possible time delays in cross analysis
   for a given set of time arrays.

   :param t1: First time array. If it is a scalar, assume regular spaced data of length
              specified by t1.
   :type t1: array_like or int
   :param t2: Second time array. If it is a scalar, assume regular spaced data of length
              specified by t2. If not given, t1 data is taken.
   :type t2: array_like or None
   :param n: The length of the output (max time delay - 1).
   :type n: int, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added. If defeined, this takes precedence over the 'n' parameter.
   :type aout: ndarray, optional

   .. rubric:: Examples

   >>> cross_count(10,n=5)
   array([10, 18, 16, 14, 12])
   >>> cross_count([1,3,6],[0,2,6],n=5)
   array([1, 3, 0, 2, 1])


.. function:: auto_count(t, n=None, aout=None)

   Culculate number of occurences of possible time delays in auto analysis
   for a given time array.

   :param t: Time array. If it is a scalar, assume regular spaced data of length specified by 't'
   :type t: array_like or int
   :param n: The length of the output (max time delay - 1).
   :type n: int, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added. If defeined, this takes precedence over the 'n' parameter.
   :type aout: ndarray, optional

   .. rubric:: Examples

   >>> auto_count(10)
   array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1])
   >>> auto_count([0,2,4,5])
   array([4, 1, 2, 1, 1, 1])


.. function:: cross_sum(f, t=None, t_other=None, axis=0, n=None, align=False, aout=None)

   Calculates sum of array, useful for normalization of correlation data.

   This function performs:
   out[k] = sum_{i,j, where  k = abs(t[i]-t_other[j])} (f[i])

   :param f: Input array
   :type f: array-like
   :param t1: Time sequence of iput array. If not given, regular-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Time sequence of the other array. If not given, t1 time sequence is assumed.
   :type t2: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Calculated sum.
   :rtype: ndarray


.. function:: cross_sum_fft(f, t, t_other=None, axis=0, n=None, aout=None)

   Calculates sum of array, useful for normalization of correlation data.

   This function is defined for irregular-spaced data only.

   See :func:`cross_sum` for details.


.. function:: auto_sum(f, t=None, axis=0, n=None, align=False, aout=None)

   Calculates sum of array, useful for normalization of autocorrelation data.

   This function performs:
   out[k] = sum_{i,j, where  k = abs(t[i]-t[j]), j >= i} (f[i]+f[j])/2.

   :param f: Input array
   :type f: array_like
   :param t1: Time sequence of iput array. If not given, regular-spaced data is assumed.
   :type t1: array-like, optional
   :param t2: Time sequence of the other array. If not given, t1 time sequence is assumed.
   :type t2: array-like, optional
   :param axis: For multi-dimensional arrays this defines computation axis (0 by default)
   :type axis: int, optional
   :param n: Maximum time delay of the output array. If not given, input data length
             is chosen.
   :type n: int, optional
   :param align: Specifies whether data is aligned in memory first, before computation takes place.
                 This may speed up computation in some cases (large n). Note that this requires
                 a complete copy of the input arrays.
   :type align: bool, optional
   :param aout: If provided, this must be zero-initiated output array to which data is
                added.
   :type aout: ndarray, optional

   :returns: **out** -- Computed sum.
   :rtype: ndarray


.. function:: auto_sum_fft(f, t, axis=0, n=None, aout=None)

   Calculates sum of array, useful for normalization of correlation data.

   This function is defined for irregular-spaced data only.

   See :func:`auto_sum` for details.


.. function:: subtract_background(data, axis=0, bg=None, return_bg=False, out=None)

   Subtracts background frame from a given data array.

   This function can be used to subtract user defined background data, or to
   compute and subtract background data.


.. function:: stats(f1, f2=None, axis=0)

   Computes statistical parameters for normalization of correlation data.

   :param f1: Fourier transform of the first video.
   :type f1: ndarray
   :param f2: Second data set (for dual video)
   :type f2: ndarray, optional
   :param axis: Axis over which to compute the statistics.
   :type axis: int, optional

   :returns: **(f1mean, f2mean), (f1var, f2var)** -- Computed mean and variance data of the input arrays.
   :rtype: (ndarray, ndarray), (ndarray, ndarray)


.. function:: acorr(f, t=None, fs=None, n=None, norm=None, method=None, align=False, axis=0, aout=None)

   Computes auto-correlation of the input signals of regular or irregular
   time - spaced data.

   If data has ndim > 1, autocorrelation is performed over the axis defined by
   the axis parameter. If 'aout' is specified the arrays must be zero-initiated.

   :param f: A complex ND array..
   :type f: array-like
   :param t: Array of integers defining frame times of the data. If not provided,
             regular time-spaced data is assumed.
   :type t: array-like, optional
   :param n: If provided, determines the length of the output. Note that 'aout' parameter
             takes precedence over 'n'.
   :type n: int, optional
   :param norm: Specifies normalization procedure 0,1,2, or 3. Default to 3, except for
                'diff' method where it default to 1.
   :type norm: int, optional
   :param method: Either 'fft' , 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param align: Whether to align data prior to calculation. Note that a complete copy of
                 the data takes place.
   :type align: bool, optional
   :param axis: Axis over which to calculate.
   :type axis: int, optional
   :param aout: Tuple of output arrays.
                For method =  'diff' : (corr, count, _, _)
                for 'corr' and 'fft' : (corr, count, squaresum, sum, _)
   :type aout: a tuple of ndarrays, optional

   :returns: * **(corr, count, squaresum, sum, _)** (*(ndarray, ndarray, ndarray, ndarray, NoneType)*) -- Computed correlation data for 'fft' and 'corr' methods,
               If norm = 3, these are all defined. For norm < 3, some may be NoneType.
             * **(diff, count, _, _)** (*(ndarray, ndarray, NoneType, NoneType)*) -- Computed difference data for 'diff' method.


.. function:: ccorr(f1, f2, t1=None, t2=None, n=None, norm=None, method=None, align=False, axis=0, f1s=None, f2s=None, aout=None)

   Computes cross-correlation of the input signals of regular or irregular
   time - spaced data.

   If data has ndim > 1, calculation is performed over the axis defined by
   the axis parameter. If 'aout' is specified the arrays must be zero-initiated.

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
   :param n: If provided, determines the length of the output. Note that 'aout' parameter
             takes precedence over 'n'.
   :type n: int, optional
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
   :param f1s: First absolute square of the input data. For norm = NORM_COMPENSATED square of the
               signal is analysed. If not given it is calculated on the fly.
   :type f1s: array-like, optional
   :param f2s: Second absolute square of the input data.
   :type f2s: array-like, optional
   :param aout: Tuple of output arrays.
                For method =  'diff' : (corr, count, sum1, sum2)
                for 'corr' and 'fft' : (corr, count, squaresum, sum1, sum2)
   :type aout: a tuple of ndarrays, optional

   :returns: * **(corr, count, squaresum, sum1, sum2)** (*(ndarray, ndarray, ndarray, ndarray, ndarray)*) -- Computed correlation data for 'fft' and 'corr' methods,
               If norm = 3, these are all defined. For norm < 3, some may be NoneType.
             * **(diff, count, sum1, sum2)** (*(ndarray, ndarray, ndarray, ndarray)*) -- Computed difference data for 'diff' method.

   .. rubric:: Examples

   Say we have two datasets f1 and f2. To compute cross-correlation of both
   datasets :

   >>> f1, f2 = np.random.randn(24,4,6) + 0j, np.random.randn(24,4,6) + 0j

   >>> data = ccorr(f1, f2, n = 16)

   Now we can set the 'out' parameter, and the results of the next dataset
   are added to results of the first dataset:

   >>> data = ccorr(f1, f2,  aout = data)

   Note that the parameter 'n' = 64 is automatically determined here, based on the
   provided 'aout' arrays.


.. function:: iccorr(data, t1=None, t2=None, n=2**5, norm=0, method='corr', count=None, chunk_size=None, thread_divisor=None, auto_background=False, viewer=None, viewer_interval=1, mode='full', mask=None, stats=False)

   Iterative version of :func:`ccorr`.



   :param data: An iterable object, iterating over dual-frame ndarray data.
   :type data: iterable
   :param t1: Array of integers defining frame times of the first data. If it is a scalar
              it defines the length of the input data
   :type t1: int or array-like, optional
   :param t2: Array of integers defining frame times of the second data. If not provided,
              regular time-spaced data is assumed.
   :type t2: array-like, optional
   :param n: Determines the length of the output.
   :type n: int, required
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param count: If given, it defines how many elements of the data to process. If not given,
                 count is set to len(t1) if that is not specified, it is set to len(data).
   :type count: int, optional
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
                cross correlation between neigbouring chunks is not computed.
   :type mode: str
   :param mask: If specifed, computation is done only over elements specified by the mask.
                The rest of elements are not computed, np.nan values are written to output
                arrays.
   :type mask: ndarray, optional
   :param stats: Whether to return stats as well.
   :type stats: bool

   :returns: **fast, slow** -- A tuple of linear_data (same as from ccorr function) and a tuple of multilevel
             data.
   :rtype: lin_data, multilevel_data


.. function:: iacorr(data, t=None, n=None, norm=0, method='corr', count=None, chunk_size=None, thread_divisor=None, auto_background=False, viewer=None, viewer_interval=1, mode='full', mask=None, stats=False)

   Iterative version of :func:`ccorr`

   :param data: An iterable object, iterating over single-frame ndarray data.
   :type data: iterable
   :param t: Array of integers defining frame times of the data. If it is a scalar
             it defines the length of the input data
   :type t: int or array-like, optional
   :param n: Determines the length of the output. Maximum value is half of the input
             length.
   :type n: int, required
   :param norm: Specifies normalization procedure 0,1,2, or 3 (default).
   :type norm: int, optional
   :param method: Either 'fft', 'corr' or 'diff'. If not given it is chosen automatically based on
                  the rest of the input parameters.
   :type method: str, optional
   :param chunk_size: Length of data chunk.
   :type chunk_size: int
   :param count: If given, it defines how many elements of the data to process. If not given,
                 count is set to len(t1) if that is not specified, it is set to len(data).
   :type count: int, optional
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
                cross correlation between neigbouring chunks is not computed.
   :type mode: str
   :param mask: If specifed, computation is done only over elements specified by the mask.
                The rest of elements are not computed, np.nan values are written to output
                arrays.
   :type mask: ndarray, optional
   :param stats: Whether to return stats as well.
   :type stats: bool

   :returns: **fast, slow** -- A tuple of linear_data (same as from ccorr function) and a tuple of multilevel
             data.
   :rtype: lin_data, multilevel_data


.. function:: take_data(data, mask)

   Selects correlation(difference) data at given masked indices.

   :param data: Data tuple as returned by `ccorr` and `acorr` functions
   :type data: tuple of ndarrays
   :param mask: A boolean frame mask array
   :type mask: ndarray

   :returns: **out** -- Same data structure as input data, but with all arrays in data masked
             with the provided mask array.
   :rtype: tuple


.. function:: normalize(data, background=None, variance=None, norm=None, mode='corr', scale=False, mask=None, out=None)

   Normalizes correlation (difference) data. Data must be data as returned
   from ccorr or acorr functions.

   Except forthe most basic normalization, background and variance data must be provided.
   Tou can use :func:`stats` to compute background and variance data.

   :param data: Input data, a length 4 (difference data) or length 5 tuple (correlation data)
   :type data: tuple of ndarrays
   :param background: Background (mean) of the frame(s) in k-space
   :type background: (ndarray, ndarray) or ndarray, optional
   :param variance: Variance of the frame(s) in k-space
   :type variance: (ndarray, ndarray) or ndarray, optional
   :param norm: Normalization type (0:baseline,1:compensation,2:bg subtract,
                3: compensation + bg subtract). Input data must support the chosen
                normalization, otherwise exception is raised. If not given it is chosen
                based on the input data.
   :type norm: int, optional
   :param mode: Representation mode: either "corr" (default) for correlation function,
                or "diff" for image structure function (image difference).
   :type mode: str, optional
   :param scale: If specified, performs scaling so that data is scaled beteween 0 and 1.
                 This works in connection with variance, which must be provided.
   :type scale: bool, optional
   :param mask: An array of bools indicating which k-values should we select. If not
                given, compute at every k-value.
   :type mask: ndarray, optional
   :param out: Output array
   :type out: ndarray, optional

   :returns: **out** -- Normalized data.
   :rtype: ndarray


