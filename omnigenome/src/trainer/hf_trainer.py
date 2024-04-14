# -*- coding: utf-8 -*-
# file: hf_trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from ... import __version__ as omnigenome_version
from ... import __name__ as omnigenome_name

from transformers import Trainer
from transformers import TrainingArguments


class HFTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }


class HFTrainingArguments(TrainingArguments):
    def __init__(self, *args, **kwargs):
        super(HFTrainingArguments, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenome_name,
            "omnigenome_version": omnigenome_version,
        }
