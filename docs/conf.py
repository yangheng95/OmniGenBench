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
    'sphinx_copybutton',
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
# }

# html_theme_options = {
#     "light_logo": "OMNIGENBENCH3.svg",
#     "dark_logo": "OMNIGENBENCH4.svg",
#     # "source_repository": "https://github.com/COLA-Laboratory/OmniGenBench",
#     # "source_branch": "master",
#     # "source_directory": "docs/",
#     # "source_view_link": "https://github.com/COLA-Laboratory/OmniGenBench/readme.md",

# }

templates_path = ['_templates']

html_theme_options = {
    "light_logo": "OMNIGENBENCH3.svg",
    "dark_logo": "OMNIGENBENCH4.svg",
    "source_edit_link": "https://github.com/COLA-Laboratory/OmniGenBench/blob/master/README.MD",
    "source_view_link": "https://cola-szhou.github.io/OmniGenBench_web",
    # "footer_icons": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/COLA-Laboratory/OmniGenBench",
    #         "html": """
    #             <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
    #                 <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    #             </svg>
    #         """,
    #         "class": "",
    #     },
    #     {
    #         "name": "Website",
    #         "url": "https://cola-szhou.github.io/OmniGenBench_web",
    #         "html": """
    #             <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
    #                 <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
    #             </svg>
    #         """,
    #         "class": "",
    #     },
    # ],
}

