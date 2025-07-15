# -*- coding: utf-8 -*-
# file: __init__.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Trainer module for genomic models.

This module provides various trainer implementations for training genomic models,
including native PyTorch training, distributed training with HuggingFace Accelerate,
and integration with HuggingFace Trainer.

Available Trainers:
------------------
- BaseTrainer: Abstract base class for all trainers
- Trainer: Native PyTorch trainer with mixed precision support
- AccelerateTrainer: Distributed training with HuggingFace Accelerate
- HFTrainer: Integration with HuggingFace Trainer ecosystem
- HFTrainingArguments: HuggingFace training arguments wrapper

Example:
--------
>>> from omnigenbench.src.trainer import Trainer
>>> trainer = Trainer(
...     model=model,
...     train_dataset=train_dataset,
...     eval_dataset=eval_dataset,
...     epochs=10,
...     batch_size=32,
...     optimizer=optimizer
... )
>>> metrics = trainer.train()
"""

from .base_trainer import BaseTrainer
from .trainer import Trainer
from .accelerate_trainer import AccelerateTrainer
from .hf_trainer import HFTrainer, HFTrainingArguments

__all__ = [
    "BaseTrainer",
    "Trainer",
    "AccelerateTrainer",
    "HFTrainer",
    "HFTrainingArguments",
]
