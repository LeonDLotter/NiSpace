#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------

project = 'NiSpace'
copyright = '2026, Leon D. Lotter'
author = 'Leon D. Lotter'

# Version
sys.path.insert(0, os.path.abspath(os.path.pardir))
import nispace 
version = nispace.__version__
release = nispace.__version__

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('../nispace'))

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting'
]

napoleon_google_docstring = True   # Turn off googledoc strings
napoleon_numpy_docstring = True     # Turn on numpydoc strings
napoleon_use_param = True
napoleon_use_rtype = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'nb_examples/ds*', 'nb_examples/_archive']
autodoc_default_options = {
    'members': True, 
    'inherited-members': True,
    #'special-members': '__init__'
    'undoc-members': True,
    'show-inheritance': True,
}
autosummary_generate = True
autodoc_typehints = 'description'

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
]
nbsphinx_codecell_lexer = 'ipython3'

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = project + ' - version ' + release
html_static_path = ['_static']
html_theme_options = {
    "repository_url": "https://github.com/leondlotter/nispace",
    "use_repository_button": True,
}

pygments_style = 'sphinx'

