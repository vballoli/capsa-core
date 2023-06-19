# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../capsa'))

import datetime 
project = 'capsa'
copyright = f'{datetime.datetime.now().year}, Themis AI Inc'
author = 'Themis AI'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_tabs.tabs",
    "nbsphinx",

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_theme_options = {
#     "dark_css_variables": {
#         "color-brand-primary": "red",
#         "color-brand-content": "#CC3333",
#         "color-admonition-background": "orange",
#     },
# }

# html_theme = 'sphinx_material'

# # Material theme options (see theme.conf for more information)
# html_theme_options = {

#     # Set the name of the project to appear in the navigation.
#     'nav_title': 'Capsa - ThemisAI',

#     # Set you GA account ID to enable tracking
#     'google_analytics_account': 'UA-XXXXX',

#     # Specify a base_url used to generate sitemap.xml. If not
#     # specified, then no sitemap will be built.
#     'base_url': 'https://themisai.io/capsa',

#     # Set the color and the accent color
#     'color_primary': 'indigo',
#     'color_accent': 'red',
#     # 'theme_color': 'AD2AB6',

#     # Set the repo location to get a badge with stats
#     'repo_url': 'https://github.com/themis-ai/capsa/',
#     'repo_name': 'Capsa',

#     # Visible levels of the global TOC; -1 means unlimited
#     'globaltoc_depth': 3,
#     # If False, expand all TOC entries
#     'globaltoc_collapse': False,
#     # If True, show hidden TOC entries
#     'globaltoc_includehidden': False,
# }

# document __init__ -- https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip)

# don't sort Sphinx output in alphabetical order
autodoc_member_order = 'bysource'
