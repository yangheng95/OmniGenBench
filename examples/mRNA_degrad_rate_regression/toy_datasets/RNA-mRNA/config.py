# -*- coding: utf-8 -*-
# file: config.py
# time: 23:04 26/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import os

import numpy as np
import torch

from omnigenbench import (
    AutoConfig,
    OmniDatasetForTokenRegression,
    OmniModelForTokenRegression,
    RegressionMetric,
)


class Dataset(OmniDatasetForTokenRegression):
    def __init__(self, dataset_name_or_path, tokenizer, max_length, **kwargs):
        super().__init__(dataset_name_or_path, tokenizer, max_length, **kwargs)

    def prepare_input(self, instance, **kwargs):
        target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        instance["sequence"] = f'{instance["sequence"]}'
        tokenized_inputs = self.tokenizer(
            instance["sequence"],
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = [instance[target_col] for target_col in target_cols]
        labels = np.concatenate(
            [
                np.array(labels),
                np.array(
                    [
                        [-100]
                        * (len(tokenized_inputs["input_ids"].squeeze()) - len(labels[0])),
                        [-100]
                        * (len(tokenized_inputs["input_ids"].squeeze()) - len(labels[0])),
                        [-100]
                        * (len(tokenized_inputs["input_ids"].squeeze()) - len(labels[0])),
                    ]
                ),
            ],
            axis=1,
        ).T
        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()
        return tokenized_inputs


# Hyperparameters
config_dict = {
    "task_name": "RNA-mRNA",
    "task_type": "token_regression",
    "label2id": None,  # For Sequence Classification
    "num_labels": 3,  # For Sequence Classification
    "epochs": 50,
    "patience": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "batch_size": 4,
    "max_length": 128,  # "max_length": 1024 for some models
    "seeds": [8946],
    "use_str": True,
    "use_kmer": True,
    "compute_metrics": [RegressionMetric().root_mean_squared_error],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json") else None,
    # "dataset_cls": Dataset,  # For your custom dataset preparation
    "dataset_cls": Dataset,
    "model_cls": OmniModelForTokenRegression,
}

bench_config = AutoConfig(config_dict)
