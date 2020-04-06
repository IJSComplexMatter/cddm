:mod:`cddm.print_tools`
=======================

.. py:module:: cddm.print_tools

.. autoapi-nested-parse::

   Message printing functions. cddm uses cddm.conf.CDDMConfig.verbose
   variable to define verbosity level and behavior of printing functions.

   verbosity levels 1 and 2 enable printing, 0 disables printing messages.



Module Contents
---------------

.. function:: print1(*args, **kwargs)

   prints level 1 messages


.. function:: print2(*args, **kwargs)

   prints level 2 messages


.. function:: print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='=')

   Call in a loop to create terminal progress bar


.. function:: print_frame_rate(n_frames, t0, t1=None, message='... processed')

   Prints calculated frame rate


.. function:: disable_prints()

   Disable message printing. Returns previous verbosity level


.. function:: enable_prints(level=None)

   Enable message printing. Returns previous verbosity level


