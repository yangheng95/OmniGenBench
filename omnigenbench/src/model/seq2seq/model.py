# -*- coding: utf-8 -*-
# file: model.py
# time: 11:40 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Sequence-to-sequence model for genomic sequences.

This module provides a sequence-to-sequence model implementation for genomic
sequences. It's designed for tasks where the input and output are both
sequences, such as sequence translation, structure prediction, or sequence
transformation tasks.
"""

from ...abc.abstract_model import OmniModel


class OmniModelForSeq2Seq(OmniModel):
    """
    This model implements a sequence-to-sequence architecture for genomic
    sequences, where the input is one sequence and the output is another
    sequence. It's useful for tasks like sequence translation, structure
    prediction, or sequence transformation. The model can be extended to implement specific seq2seq tasks by
    overriding the forward, predict, and inference methods.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the sequence-to-sequence model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
