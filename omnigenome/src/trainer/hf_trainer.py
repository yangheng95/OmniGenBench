# -*- coding: utf-8 -*-
# file: hf_trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
HuggingFace trainer integration for OmniGenome models.

This module provides HuggingFace trainer wrappers for OmniGenome models,
enabling seamless integration with the HuggingFace training ecosystem
while maintaining OmniGenome-specific functionality.
"""

from transformers import Trainer
from transformers import TrainingArguments

from ... import __name__ as omnigenome_name
from ... import __version__ as omnigenome_version


class HFTrainer(Trainer):
    """
    HuggingFace trainer wrapper for OmniGenome models.
    
    This class extends the HuggingFace Trainer to include OmniGenome-specific
    metadata and functionality while maintaining full compatibility with the
    HuggingFace training ecosystem.
    
    Attributes:
        metadata: Dictionary containing OmniGenome library information
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the HuggingFace trainer wrapper.
        
        Args:
            *args: Positional arguments passed to the parent Trainer
            **kwargs: Keyword arguments passed to the parent Trainer
        """
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }


class HFTrainingArguments(TrainingArguments):
    """
    HuggingFace training arguments wrapper for OmniGenome models.
    
    This class extends the HuggingFace TrainingArguments to include
    OmniGenome-specific metadata while maintaining full compatibility
    with the HuggingFace training ecosystem.
    
    Attributes:
        metadata: Dictionary containing OmniGenome library information
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the HuggingFace training arguments wrapper.
        
        Args:
            *args: Positional arguments passed to the parent TrainingArguments
            **kwargs: Keyword arguments passed to the parent TrainingArguments
        """
        super(HFTrainingArguments, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }
