# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GP-melt'
copyright = '2024, Charles Le Losq, Clément Ferraina'
author = 'Charles Le Losq, Clément Ferraina'
release = '0.1'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosectionlabel',
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

# -- Project information -----------------------------------------------------

project = 'GP-visc'
copyright = '2024, Charles Le Losq et co.'
author = 'Charles Le Losq, Clément Ferraina, Charles-Édouard Boukaré'

# The full version, including alpha/beta/rc tags
release = '0.5.0'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
