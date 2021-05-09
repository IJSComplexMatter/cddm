# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.split(__file__)[0]))

# -- Project information -----------------------------------------------------

project = 'cddm'
copyright = '2020, Andrej Petelin'
author = 'Andrej Petelin'

# The full version, including alpha/beta/rc tags
release = '0.4.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.


extensions = ['sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
	"sphinx.ext.doctest",
        'sphinx.ext.inheritance_diagram',    
	'autoapi.extension',
	'matplotlib.sphinxext.plot_directive'
    ]

doctest_path = [os.path.abspath("examples")] 
doctest_global_setup = '''
try:
    import numpy as np
    from cddm.fft import * 
    from cddm.core import * 
    from cddm.decorators import * 
    from cddm.multitau import * 
    from cddm.window import *
    from cddm.video import *
    from cddm.sim import * 
except ImportError:
	pass
'''

    
plot_working_directory = "examples"
#plot_include_source = True

autoapi_keep_files = True
    
napoleon_numpy_docstring = True

autoapi_dirs = ['../cddm']
autoapi_options = ['members', 'undoc-members', 'show-inheritance', 'special-members']
autoapi_options = ['members', 'show-inheritance']
autoapi_ignore = ["*test_*.py"]

numfig = True

# custom matplotlib plot_template

if sys.argv[2] in ('latex', 'latexpdf'):
    plot_template = """
{% for img in images %}
.. figure:: {{ build_dir }}/{{ img.basename }}.pdf
    {%- for option in options %}
    {{ option }}
    {% endfor %}
    
    \t{{caption}}
{% endfor %}
"""

else:
    plot_template = """
{% for img in images %}

.. figure:: {{ build_dir }}/{{ img.basename }}.png
    {%- for option in options %}
    {{ option }}
    {% endfor %}

    \t{% if html_show_formats and multi_image -%}
    (
    {%- for fmt in img.formats -%}
    {%- if not loop.first -%}, {% endif -%}
    `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    )
    {%- endif -%}

    {{ caption }} {% if source_link or (html_show_formats and not multi_image) %} (
{%- if source_link -%}
`Source code <{{ source_link }}>`__
{%- endif -%}
{%- if html_show_formats and not multi_image -%}
    {%- for img in images -%}
    {%- for fmt in img.formats -%}
        {%- if source_link or not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
    {%- endfor -%}
    {%- endfor -%}
{%- endif -%}
)
{% endif %}
{% endfor %}
"""




# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
