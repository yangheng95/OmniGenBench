# -*- coding: utf-8 -*-
# file: config.py
# time: 23:04 26/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import os

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig, OmniGenomeDatasetForSequenceClassification, OmniGenomeModelForMultiLabelSequenceClassification,
)


# Hyperparameters
config_dict = {
    "task_name": "chromatin_access.brachypodium_distachyon",
    "task_type": "multi_label_sequence_classification",
    "label2id": None,  # For Sequence Classification
    "num_labels": 9,  # For Sequence Classification
    "epochs": 20,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "batch_size": 32,
    "max_length": 512,  # "max_length": 1024 for some models
    "seeds": [45, 46, 47],
    "compute_metrics": ClassificationMetric(ignore_y=-100, average="macro").f1_score,
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json")
    else None,
    "dataset_cls": OmniGenomeDatasetForSequenceClassification,  # For your custom dataset preparation
    "model_cls": OmniGenomeModelForMultiLabelSequenceClassification,
}

bench_config = AutoBenchConfig(config_dict)