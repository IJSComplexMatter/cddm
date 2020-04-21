Introduction
============

``cddm`` is a python package for analysis of cross-differential dynamic microscopy (cross-DDM) experiments. You can also use it for a single-camera (DDM) analysis, or a general purpose correlation analysis. Regular and irregular time-spaced data is supported. The package is hosted at GitHub `repository`_.

License
-------

``cddm`` is released under MIT license so you are able to use it freely. I kindly ask you to cite the package if you use it. See the DOI badge in the `repository`_.

Highlights
----------

* Easy-to-use interface.
* Efficient computation (numba optimized), multiple-tau algorithm.
* In-memory and out-of-memory computation of cross- auto-correlation functions.
* Data visualizer, and k-averaging.

Status and limitations
----------------------

``cddm``  package was developed mainly for the analysis of cross-DDM experiment, but it can be used for standard DDM analysis or any kind of correlation analysis. The package is fully operational. Currently, the work is focused on documenting the package and writing examples, and unittests. The API is stable, and is not expected to change in the future release version.

.. _repository: https://github.com/IJSComplexMatter/cddm




