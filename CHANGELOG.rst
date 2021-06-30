Release notes
-------------

V0.4.0 (Jul 2 2021)
+++++++++++++++++++

With this release we add new memory layout used in iccor_multi function. 
Combined with some other tweaks, it improves the computation speed. 

New features
////////////

* added conf.set_num_threads.

V0.3.0 (Mar 14 2021)
++++++++++++++++++++

This release partially breaks backward compatibility. Most users will not notice any difference, but be aware that the naming of normalization schemes have changed slightly. With this release we define NORM_STRUCTURED normalization, which was previously called NORM_COMPENSATED. The NORM_COMPENSATED now marks an experimental (photon-correlation-like compensated normalization scheme). Users need to adopt their code by renaming the NORM_COMPENSATED with NORM_STRUCTURED.
Also, the flag values have changed. Therefore, if you used integer values for norm flags, and you did not setup the flags using the provided helper functions then you need to revise your code.

New features
////////////

* added functions to sim.py to simulate ADC signal conversion, readout noise and shot noise.
* weight_from_data, returns weight for the structured normalization. In previous versions
  it was for the standard normalization.
* added `norm.py` module and moved all normalization related functions here.
* added functions for correlation data error estimation.
* added `normalize` argument to `show_diff` function.
* added the `weight` argument to `normalize` function.
* added a threaded version of `play` function called :func:`.video.play_threaded`.

Enhancements
////////////

* `rfft2` now uses `rfftn_numpy` when using mkl_fft 
* `rfft2` now accepts extra keyword argument that can be used to pass extra arguments for
  the underlying fft library.

Bug fixes
/////////

* removed noise argument in simulation functions because shot noise was not implemented correctly.
  Shot noise is now simulated during ADC conversion. 
* Renamed `particles` argument to `num_particles` in `brownian_particles` function.
* Fixes segfaults which were caused by numba's mixup of cached data.


V0.2.0 (May 6 2020)
+++++++++++++++++++

New features
////////////

* Added NORM_WEIGHTED flag for weighted normalization. Added :func:`.core.norm_flags`.
* Added `avg.py` module and a few new functions for data averaging.
* Added a few functions in multitau.py for effective count calculation for error estimation in example `cross_correlate_error.py`
* Updated :class:`.viewer.CorrViewer` to implement new normalization type.
* Updated :func:`.core.normalize` and :func:`.multitau.normalize_multi` to implement new normalization type.
* Structured examples into a package.
* Add new examples.

Bug fixes
/////////

* Fixes a bug in calculation with binning = 2 (random selection). 
* Fixes a bug in calculating merge_multilevel(data, mode = "half").

V0.1.2 (Apr 21 2020)
++++++++++++++++++++

Bugfix release, updated documentation.

V0.1.0 (Apr 21 2020)
++++++++++++++++++++

Initial release.