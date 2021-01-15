:mod:`cddm.norm`
================

.. py:module:: cddm.norm

.. autoapi-nested-parse::

   Normalization helper functions



Module Contents
---------------





.. data:: NORM_STANDARD
   :annotation: = 1

   standard normalization flag


.. data:: NORM_STRUCTURED
   :annotation: = 2

   structured normalization flag


.. data:: NORM_WEIGHTED
   

   weighted normalization flag


.. data:: NORM_SUBTRACTED
   :annotation: = 4

   background subtraction flag


.. data:: NORM_COMPENSATED
   :annotation: = 8

   compensated normalization flag


.. function:: norm_flags(structured=False, subtracted=False, weighted=False, compensated=False)

   Return normalization flags from the parameters.

   :param structured: Whether to set the STRUCTURED normalization flag.
   :type structured: bool
   :param subtracted: Whether to set SUBTRACTED normalization flag.
   :type subtracted: bool
   :param weighted: Whether to set WEIGHTED normalization flags.
   :type weighted: bool
   :param compensated: Whether to set COMPENSATED normalization flag.
   :type compensated: bool

   :returns: **norm** -- Normalization flags.
   :rtype: int

   .. rubric:: Examples

   >>> norm_flags(structured = True)
   2
   >>> norm_flags(compensated = True)
   13


.. function:: scale_factor(variance, mask=None)

   Computes the normalization scaling factor from the variance data.

   You can divide the computed correlation data with this factor to normalize
   data between (0,1) for correlation mode, or (0,2) for difference mode.

   :param variance: A variance data (as returned from :func:`.stats`)
   :type variance: (ndarray, ndarray) or ndarray
   :param mask: A boolean mask array, if computation was performed on masked data,
                this applys mask to the variance data.
   :type mask: ndarray

   :returns: **scale** -- A scaling factor for normalization
   :rtype: ndarray


.. function:: noise_delta(variance, mask=None, scale=True)

   Computes the scalled noise difference from the variance data.

   This is the delta parameter for weighted normalization.

   :param variance: A variance data (as returned from :func:`.stats`)
   :type variance: (ndarray, ndarray)
   :param mask: A boolean mask array, if computation was performed on masked data,
                this applys mask to the variance data.
   :type mask: ndarray

   :returns: **delta** -- Scalled delta value.
   :rtype: ndarray


.. function:: weight_from_data(corr, delta=0.0, scale_factor=1.0, mode='corr', pre_filter=True)

   Computes weighting function for weighted normalization.

   :param corr: Correlation (or difference) data
   :type corr: ndarray
   :param scale_factor: Scaling factor as returned by :func:`.core.scale_factor`. If not provided,
                        corr data must be computed with scale = True option.
   :type scale_factor: ndarray
   :param mode: Representation mode of the data, either 'corr' (default) or 'diff'
   :type mode: str
   :param pre_filter: Whether to perform denoising and filtering. If set to False, user has
                      to perform data filtering.
   :type pre_filter: bool

   :returns: **out** -- Weight data for weighted sum calculation.
   :rtype: ndarray


.. function:: normalize(data, background=None, variance=None, norm=None, mode='corr', scale=False, mask=None, weight=None, ret_weight=False, out=None)

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
   :param weight: If you wish to specify your own weight for weighted normalization, you
                  must provide it here, otherwise it is computed from the data (default).
   :type weight: ndarray, optional
   :param ret_weight: Whether to return weight (when calculating weighted normalization)
   :type ret_weight: bool, optional
   :param out: Output array
   :type out: ndarray, optional

   :returns: * **out** (*ndarray*) -- Normalized data.
             * **out, weight** (*ndarray, ndarray*) -- Normalized data and weight if 'ret_weight' was specified


.. function:: take_data(data, mask)

   Selects correlation(difference) data at given masked indices.

   :param data: Data tuple as returned by `ccorr` and `acorr` functions
   :type data: tuple of ndarrays
   :param mask: A boolean frame mask array
   :type mask: ndarray

   :returns: **out** -- Same data structure as input data, but with all arrays in data masked
             with the provided mask array.
   :rtype: tuple


