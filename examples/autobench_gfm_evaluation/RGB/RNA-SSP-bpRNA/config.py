# -*- coding: utf-8 -*-
# file: config.py
# time: 23:04 26/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import os

from omnigenbench import (
    ClassificationMetric,
    AutoConfig,
    OmniDatasetForTokenClassification,
    OmniModelForTokenClassification,
)

label2id = {"(": 0, ")": 1, ".": 2}

# Hyperparameters
config_dict = {
    "task_name": "RNA-SSP-bpRNA",
    "task_type": "token_classification",
    "label2id": label2id,  # For Sequence Classification
    "num_labels": None,  # For Sequence Classification
    "epochs": 50,
    "patience": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "batch_size": 4,
    "max_length": 1024,  # "max_length": 1024 for some models
    "seeds": [8946],
    "use_str": False,
    "use_kmer": True,
    "compute_metrics": [ClassificationMetric(ignore_y=-100, average="macro").f1_score,
                        ClassificationMetric(ignore_y=-100).matthews_corrcoef],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json") else None,
    # "dataset_cls": Dataset,  # For your custom dataset preparation
    "dataset_cls": OmniDatasetForTokenClassification,
    "model_cls": OmniModelForTokenClassification,
}

bench_config = AutoConfig(config_dict)
