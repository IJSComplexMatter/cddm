Installation
============

The source code is hosted at GitHub. You can install latest stable release with::

    $ pip install cddm

Or you can clone or download the latest development code from the `repository`_ and run::

    $ python setup.py install

which should install the package, provided that all requirements are met.

Requirements
------------

Prior to installing, you should have a working python 3.x environment consisting of:

* numba
* numpy
* matplotlib

To install these it is best to go with one of the python distributions, e.g. `anaconda`_, or any other python distribution that comes shipped with the above packages. 

.. note::
  
    It is important that you have a recent enough version of numba installed. The package was tested with numba 0.45 and above, older versions might work as well, but it was not tested.

Optionally, instead of using matplotlib, for faster visualization of videos in real-time, you may install and use one of these two libraries:

* cv2
* pyqtgraph

For faster computation of FFTs you may install and use one of these two libraries.

* mkl_fft
* pyfftw

Installing in Anaconda
----------------------

Make sure you have the required packages::

    $ conda install numba matplotlib numpy
    $ pip install cddm

Optionally, for faster FFT computation, you can install `mkl_fft`::

    $ conda install mkl_fft

.. _repository: https://github.com/IJSComplexMatter/cddm
.. _numba: http://numba.pydata.org
.. _anaconda: https://www.anaconda.com
.. _mkl_fft: https://github.com/IntelPython/mkl_fft