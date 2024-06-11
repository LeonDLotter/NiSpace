#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------

project = 'NiSpace'
copyright = '2024, Leon D. Lotter'
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
    'sphinx_gallery.gen_gallery'
]
napoleon_google_docstring = False   # Turn off googledoc strings
napoleon_numpy_docstring = True     # Turn on numpydoc strings
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_default_options = {'members': True, 'inherited-members': True}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
