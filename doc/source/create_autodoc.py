"""This script generates  API rst file and copies data files from source
path
"""


import os, glob, re, shutil
from setuptools import find_packages

#print 'Copying stress_strain.py'
#shutil.copy(os.path.join('..','..','labtools','experiment','stress_strain.py'),
#            os.path.join('examples','stress_strain.py'))

#SKIP = ['labtools.experiment.instr','labtools.dde', 'labtools.instr','labtools.test']
SKIP = []
#PACKAGES = ('stretcher','stretcher.standa', 'stretcher.mantracourt', 'stretcher.utils', 'stretcher.experiment')
PACKAGES = [p for p in find_packages('../..') if not p.endswith('_test') and not p in SKIP]
BASE_PACKAGE = PACKAGES[0]
SUBPACKAGES = PACKAGES[1:]

#name of the main index file for autodoc
INDEX = 'api.rst'
#which modules to skip
#SKIP = []#['USMCDLL']

AUTOMODULE = \
"""
.. automodule:: %(name)s
   :members:
   :undoc-members:
   :show-inheritance:
       
"""

#module text for autodoc
RST_MOD = \
"""
:mod:`%(relname)s`%(synopsis)s
%(line)s  

""" + AUTOMODULE


#index file header. toctree is extended by rst names
HEADER = \
""".. _API:  
    
===
API
===

This is a ddm package API. A detailed function definition along with some usage examples are provided.
For a more general introduction see :ref:`introduction` and for a quick-start guide see :ref:`quickstart`.
"""

def parse_module(module, package, section = '=', subsection = '-'):
    with open(module) as fm:
        folder,fname = os.path.split(module)
        base, ext = os.path.splitext(fname)
        if base == '__init__' or not base.startswith('_'):
            data = fm.read()
            try:
                synopsis = ' -- ' + re.search(':synopsis:(.+)',data).group(1).strip()
            except AttributeError: 
                synopsis = ''
                
            if base == '__init__':
                packagelist = package.split('.')
            else:
                packagelist = package.split('.')+[base]       
            relname = '.'.join(packagelist[1:])
            name = '.'.join(packagelist)
            if relname != '':
                if base == '__init__':
                    line = section*(len(relname) +len(synopsis)+ 7) # :mod: and ticks = 7
                    return RST_MOD % {'relname': relname, 'name' : name, 'line' : line, 'synopsis' : synopsis}  
                    #line = '-'*(len(relname) +len(synopsis)+ 7) # :mod: and ticks = 7
                    #return RST_MOD % {'relname': relname, 'name' : name, 'line' : line, 'synopsis' : synopsis}  
                else:
                    line = subsection*(len(relname) +len(synopsis)+ 7) # :mod: and ticks = 7
                    return RST_MOD % {'relname': relname, 'name' : name, 'line' : line, 'synopsis' : synopsis}  
            #then we are parsing __init__ of the base package
            else:
                return ''
        else:
            return ''


print 'opening file %s' % INDEX                
with open(INDEX,'w') as fi:
    fi.write(HEADER)
    fi.write('\n' + AUTOMODULE % {'name' : BASE_PACKAGE})

    modules = glob.glob('../../%s/*.py' % BASE_PACKAGE)    
    for module in sorted(modules):
        print 'parsing file %s' % module
        fi.write(parse_module(module, BASE_PACKAGE, subsection = '='))
        
    print 'Parsing subpackages'
    for subpackage in SUBPACKAGES:
        modules = glob.glob('../../%s/*.py' % subpackage.replace('.','/'))
        rst = os.path.join(BASE_PACKAGE, '.'.join(subpackage.split('.')[1:]) + '.rst')
        
        for module in sorted(modules):
            print 'parsing file %s' % module
            fi.write(parse_module(module, subpackage))
