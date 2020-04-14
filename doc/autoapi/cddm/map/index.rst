:mod:`cddm.map`
===============

.. py:module:: cddm.map

.. autoapi-nested-parse::

   Data mapping and k-averaging functions.



Module Contents
---------------

.. function:: sector_indexmap(kmap, anglemap, angle=0.0, sector=30.0, kstep=1.0)

   Builds indexmap array of integers ranging from -1 (invalid data)
   and positive integers. Each non-negative integer is a valid k-index computed
   from sector parameters.

   :param kmap: Size of wavevector at each (i,j) indices in the rfft2 ouptut data.
   :type kmap: ndarray
   :param anglemap: Angle of the wavevector at each (i,j) indices in the rfft2 ouptut data.
   :type anglemap: ndarray
   :param angle: Mean angle of the sector in degrees (-90 to 90).
   :type angle: float
   :param sector: Width of the sector in degrees (between 0 and 180)
   :type sector: float
   :param kstep: k resolution in units of minimum step size.
   :type kstep: float, optional

   :returns: **map** -- Ouput array of non-zero valued k-indices where data is valid, -1 elsewhere.
   :rtype: ndarray


.. function:: rfft2_kangle(kisize=None, kjsize=None, shape=None)

   Build k,angle arrays based on the size of the fft.

   :param kisize: i-size of the cropped rfft2 data.
   :type kisize: int
   :param kjsize: j-size of the cropped rfft2 data
   :type kjsize: int
   :param shape: Shape of the original data. This is used to calculate step size. If not
                 given, rectangular data is assumed (equal steps).
   :type shape: (int,int)


.. function:: k_select(data, angle, sector=5, kstep=1, k=None, shape=None, mask=None)

   k-selection and k-averaging of normalized (and merged) correlation data.

   This function takes (...,i,j,n) correlation data and performs k-based selection
   and averaging of the data.

   :param deta: Input correlation data of shape (...,i,j,n)
   :type deta: array_like
   :param angle: Angle between the j axis and the direction of k-vector
   :type angle: float
   :param sector: Defines sector angle, k-data is avergaed between angle+sector and angle-sector angles
   :type sector: float, optional
   :param kstep: Defines an approximate k step in pixel units
   :type kstep: float, optional
   :param k: If provided, only return at a given k value (and not at all non-zero k values)
   :type k: float or list of floats, optional
   :param shape: Shape of the original video frame. If shape is not rectangular, it must
                 be provided.
   :type shape: tuple
   :param mask: A boolean array indicating which data elements were computed.
   :type mask: ndarray

   :returns: **out** -- If k s not defined, this is an iterator that yields a tuple of (k_avg, data_avg)
             of actual (mean) k and averaged data. If k is a list of indices, it returns
             an iterator that yields a tuple of (k_avg, data_avg) for every non-zero data
             if k is an integer, it returns a tuple of (k_avg, data_avg) for a given k-index.
   :rtype: iterator, or tuple


