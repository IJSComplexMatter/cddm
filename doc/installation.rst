Installation
============

The code is hosted on GitHub. No official release exists yet, so you have to clone or download the latest development code from the `repository`_ and run::

    $ python setup.py install

which should install the package, provided that all requirements are met.

Requirements
------------

Prior to installing, you should have a working python 2.7 or 3.x environment consisting of:

* numba
* numpy
* matplotlib

To install these it is best to go with one of the python distributions, e.g. `anaconda`_, or any other python distribution that comes shipped with the above packages. 

.. note::
  
    It is important that you have recent enough version of numba installed.


Installing in Anaconda
----------------------

After you have downloaded the package, open the terminal (command prompt) `cd` to the downloaded source code and run::

    $ conda install numba matplotlib numpy
    $ python setup.py install

Optionally, for faster FFT computation, you can install `mkl_fft`::

    $ conda install mkl_fft


.. _repository: https://github.com/IJSComplexMatter/cddm
.. _numba: http://numba.pydata.org
.. _anaconda: https://www.anaconda.com
.. _mkl_fft: https://github.com/IntelPython/mkl_fft