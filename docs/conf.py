# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# set of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = ''
copyright = '2024-2025'
author = 'YANG, HENG'

# The full version, including alpha/beta/rc tags
# release = '0.3.0alpha'
# The short X.Y version
# version = '0.3.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.viewcode',
    'sphinx_design', 
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
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
html_theme = 'furo'
# html_theme = 'sphinx_nefertiti'
# html_theme = 'pydata_sphinx_theme'
# extensions.append("pydata_sphinx_theme")
# html_theme = 'sphinx_book_theme'

# extensions.append("sphinx_wagtail_theme")
# html_theme = 'sphinx_wagtail_theme'
# html_theme = "furo"
# html_theme = 'sphinxawesome_theme'



# Add custom CSS for better styling
html_static_path = ['_static']


# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
# napoleon_numpy_docstring = True
napoleon_use_ivar = True
autosectionlabel_prefix_document = True
napoleon_use_rtype = False

html_title = ""
# html_logo = '_static/OMNIGENBENCH.png' 

html_css_files = [
    'custom.css',
]


# html_context = {
#     "github_user": "COLA-Laboratory",
#     "github_repo": "OmniGenBench",
#     "github_version": "master", # 你的分支名
#     "doc_path": "docs/",      # 你的文档源文件目录
#     "sidebar_links": {
#         "GitHub Repository": "https://github.com/COLA-Laboratory/OmniGenBench",
#         # 你还可以添加其他链接
#         # "Issue Tracker": "https://github.com/COLA-Laboratory/OmniGenBench/issues",
#     },
# }

html_theme_options = {
    "light_logo": "OMNIGENBENCH3.svg",
    "dark_logo": "OMNIGENBENCH4.svg",
    # 这个选项本身不会添加图标，但保留它是个好习惯
    "source_repository": "https://github.com/COLA-Laboratory/OmniGenBench",
    "source_branch": "master",
    "source_directory": "docs/",
}
