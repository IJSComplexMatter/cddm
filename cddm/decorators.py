"""Decorators"""
from __future__ import absolute_import, print_function, division

from functools import wraps

class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator (doc_inherit)
    

    Example
    -------
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
    
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit 


def skip_runtime_error(f):
    """A decorator to disable messages due to runtime error in matplotlib"""
    def _f(event):
        try:
            return f(event)
        except RuntimeError:
            pass
    return _f

import warnings
import functools

def deprecated(info = ""):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def _deprecated(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Call to deprecated function {}. {}".format(func.__name__, info),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)
        return new_func
    return _deprecated

##
#if __name__ == "__main__":
#    import doctest
#    doctest.testmod()