# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:53 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

"""
OmniGenBench - Alias package for omnigenome
============================================

This package provides the same functionality as omnigenome but with the omnigenbench name.
All imports are redirected to the omnigenome package.
"""

# Import everything from omnigenome to maintain compatibility
from omnigenbench import *

# Override package metadata to reflect omnigenbench
__name__ = "omnigenome"
from omnigenbench import __version__
from omnigenbench import __author__
from omnigenbench import __email__
from omnigenbench import __license__
