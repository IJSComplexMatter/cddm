Release notes
-------------

V0.3.0 (In development)
+++++++++++++++++++++++

New features
////////////

* added functions to sim.py to simulate ADC signal conversion, readout noise and shot noise.
* weight_from_data, returns weight for the compensated normalization. In previous versions
  it was for the subtracted normalization.
* added `norm.py` module and moved all normalization related functions here.
* added functions for correlation data error estimation.

Bug fixes
/////////

* removed noise argument in simulation functions because shot noise was not implemented correctly.


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