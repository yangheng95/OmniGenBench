# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# set of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'OmniGenBench'
copyright = '2024-2025'
author = 'YANG, HENG'

# The full version, including alpha/beta/rc tags
release = '0.3.0alpha'
# The short X.Y version
version = '0.3.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

autosectionlabel_prefix_document = True

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# html_theme = 'sphinx_rtd_theme'
# html_theme = 'furo'
# html_theme = 'sphinx_nefertiti'
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'sphinx_book_theme'

# html_permalinks_icon = '<span>#</span>'
# html_theme = 'sphinxawesome_theme'

extensions.append("sphinx_wagtail_theme")
html_theme = 'sphinx_wagtail_theme'


# Add custom CSS for better styling
html_static_path = ['_static']

# Custom CSS to improve visual effects
# html_css_files = [
#     'custom.css',
# ]

html_theme_options = {
    "logo": "og_favicon.png",
    "project_name": "OmniGenBench Documentation",
    "header_links": "Home|https://cola-szhou.github.io/OmniGenBench_web/,GitHub|https://github.com/COLA-Laboratory/OmniGenBench",
}

# Version configuration for ReadTheDocs theme
html_context = {
    "display_github": True,
    "github_user": "COLA-Laboratory",
    "github_repo": "OmniGenBench",
    "github_version": "master",
    "conf_py_path": "/docs/",
    "source_suffix": ".rst",
    "github_url": "https://github.com/COLA-Laboratory/OmniGenBench",
    "github_banner": True,
    "display_version": False,
}

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
