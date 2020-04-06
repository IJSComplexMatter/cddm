Introduction
============

``cddm`` is a package for analysis of cross-differential dynamic microscopy experiments
and for standard DDM experiments. It consists of tools for the analysis of regular
or irregular time-spaced data. 


.. note::

   This package is still in its early stage of development, so it should be considered experimental. No official release exists yet! The package and the documentation is actively being worked on, so please stay tuned. The core functionality has been defined and the package is ready for use and testing, but much needs to be done to documentation and code cleanup, etc.. Expect also some minor API changes in the near future.

License
-------

``dtmm`` will be released under MIT license so you will be able to use it freely, however, I will ask you to cite the *... some future paper*. In the mean time you can play with the current development version freely, but i kindly ask you not to redistribute the code or  publish data obtained with this package. **Please wait until the package is officially released!**

Highlights
----------

* Easy-to-use interface.
* Efficient computation (numba optimized), multitau algorithm
* In-memory and out-of-memory computation of cross- auto-correlation functions
* Data visualizer, and k-averaging

   
Status and limitations
----------------------

``cddm`` is a young (experimental) project. The package was developed mainly for calculation of cross correlation function in cross-DDM experiment, but it can be used
for standard DDM analysis. The package is fully operational. Currently work is focused on documenting the package and working on examples and tutorials.

For now, inspect the examples in the examples/ folder of the source at `repository`_

.. _repository: https://github.com/IJSComplexMatter/cddm




