# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'gpvisc'
copyright = '2024, Charles Le Losq, Clément Ferraina, Paolo Sossi, Charles-Edouard Boukaré'
author = 'Charles Le Losq, Clément Ferraina, Paolo Sossi, Charles-Edouard Boukaré'
release = '0.3.3'

#import os
#import sys
#sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosectionlabel',
              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'sphinx.ext.githubpages'
              ]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
