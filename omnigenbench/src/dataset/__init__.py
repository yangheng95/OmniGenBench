# -*- coding: utf-8 -*-
# file: __init__.py
# time: 22:33 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
This package contains dataset-related modules.
"""

# Original torch-based datasets (for backward compatibility)
from .omni_dataset import (
    OmniDatasetForTokenClassification,
    OmniDatasetForSequenceClassification,
    OmniDatasetForTokenRegression,
    OmniDatasetForSequenceRegression,
    OmniDatasetForMultiLabelClassification,
)

__all__ = [
    "OmniDatasetForTokenClassification",
    "OmniDatasetForSequenceClassification",
    "OmniDatasetForTokenRegression",
    "OmniDatasetForSequenceRegression",
    "OmniDatasetForMultiLabelClassification",
]
