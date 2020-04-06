:mod:`cddm.conf`
================

.. py:module:: cddm.conf

.. autoapi-nested-parse::

   Configuration and constants



Module Contents
---------------

.. function:: get_home_dir()

   Return user home directory


.. data:: HOMEDIR
   

   Users home directory


.. data:: CDDM_CONFIG_DIR
   

   Config directory where cddm.ini lives


.. data:: NUMBA_CACHE_DIR
   

   Numba cache directory, where compiled functions are written


.. data:: CONF
   

   Configuration file filename


.. data:: NUMBA_CACHE
   :annotation: = False

   Specifiess whether we use numba caching or not


.. function:: is_module_installed(name)

   Checks whether module with name 'name' is istalled or not


.. function:: detect_number_of_cores()

   detect_number_of_cores()

   Detect the number of cores in this system.

   :returns: **out** -- The number of cores in this system.
   :rtype: int


.. function:: disable_mkl_threading()

   Disables mkl threading.


.. function:: enable_mkl_threading()

   Enables mkl threading.


.. py:class:: CDDMConfig

   Bases: :class:`object`

   DDMM settings are here. You should use the set_* functions in the
   conf.py module to set these values


.. function:: print_config()

   Prints all compile-time and run-time configurtion parameters and settings.


.. function:: set_verbose(level)

   Sets verbose level (0-2) used by compute functions.


.. function:: set_cv2(level)

   Enable/Disable cv2 rendering.


.. function:: set_nthreads(num)

   Sets number of threads used by fft functions.


.. function:: set_fftlib(name='numpy.fft')

   Sets fft library. Returns previous setting.


