# -*- coding: utf-8 -*-
# file: model.py
# time: 11:40 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from ...abc.abstract_model import OmniModel


class OmniModelForSeq2Seq(OmniModel):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
