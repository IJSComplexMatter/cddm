:mod:`cddm.decorators`
======================

.. py:module:: cddm.decorators

.. autoapi-nested-parse::

   Decorators



Module Contents
---------------

.. py:class:: DocInherit(mthd)

   Bases: :class:`object`

   Docstring inheriting method descriptor

   The class itself is also used as a decorator (doc_inherit)

   .. rubric:: Example

   >>> class Foo(object):
   ...     def foo(self):
   ...         "Frobber"
   ...         pass

   >>> class Bar(Foo):
   ...     @doc_inherit
   ...     def foo(self):
   ...         pass

   >>> Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
   True


.. function:: skip_runtime_error(f)

   A decorator to disable messages due to runtime error in matplotlib


