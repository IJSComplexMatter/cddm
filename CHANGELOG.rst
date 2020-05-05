Release notes
-------------

V0.2.0 (May 6 2020)
+++++++++++++++++++

New features
////////////

* Added NORM_WEIGHTED flag for weighted normalization. Added :func:`.core.norm_flags`.
* Added avg.py module and a few new functions for data averaging.
* Updated :class:`.viewer.CorrViewer` to implement new normalization type.
* Updated :func:`.core.normalize` and :func:`.core.normalize_multi` to implement new normalization type.
* Structured examples into a package.
* Add new examples

Bug fixes
/////////

* Fixes a bug in calculation with binning = 2 (random selection). 
* Fixes a bug in calculating merge_multilevel(data, mode = "half")

V0.1.2 (Apr 21 2020)
++++++++++++++++++++

Bugfix release, updated documentation.

V0.1.0 (Apr 21 2020)
++++++++++++++++++++

Initial release.