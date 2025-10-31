# -*- coding: utf-8 -*-
# file: __init__.py
# time: 18:27 11/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
This package contains modules for the model hub.
"""

# Import robust downloading functions (optional, with fallback)
try:
    from .hf_download import (
        download_from_hf_hub,
        verify_download_integrity,
        list_hf_repo_files,
        get_model_info,
        download_file_from_hf_hub,
    )

    __all_download__ = [
        "download_from_hf_hub",
        "verify_download_integrity",
        "list_hf_repo_files",
        "get_model_info",
        "download_file_from_hf_hub",
    ]
except ImportError:
    __all_download__ = []

__all__ = __all_download__
