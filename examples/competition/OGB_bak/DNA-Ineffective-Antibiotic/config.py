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

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig,
    OmniGenomeDatasetForSequenceClassification,
    OmniGenomeModelForSequenceClassification,
)


class Dataset(OmniGenomeDatasetForSequenceClassification):
    def prepare_input(self, instance, **kwargs):
        sequence = (
            instance.get("seq", None)
            if "seq" in instance
            else instance.get("sequence", None)
        )

        tokenized_inputs = self.tokenizer(
            sequence,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        )
        try:
            labels = instance.get("labels", None) if instance.get("labels", None) else instance.get("label", None)
            labels = torch.tensor(self.label2id[labels], dtype=torch.int64)
            tokenized_inputs["labels"] = labels
        except:
            pass # Handle the case where labels are not present

        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        return tokenized_inputs

label2id = {
    "multidrug": 0,
    "beta_lactam": 1,
    "bacitracin": 2,
    "polymyxin": 3,
    "aminoglycoside": 4,
    "glycopeptide": 5,
    "macrolide-lincosamide-streptogramin": 6,
    "quinolone": 7,
    "tetracycline": 8,
    "chloramphenicol": 9,
    "fosfomycin": 10,
    "sulfonamide": 11,
    "trimethoprim": 12,
    "unclassified":13
}

# Hyperparameters
config_dict = {
    "task_name": "DNA-Ineffective-Antibiotic",
    "task_type": "sequence_classification",
    "label2id": label2id,  # For Sequence Classification
    "num_labels": None,  # For Sequence Classification
    "epochs": 50,
    "patience": 5,
    "learning_rate": 2e-5,
    "weight_decay": 0,
    "batch_size": 4,
    "max_length": 1024,  # "max_length": 1024 for some models
    "seeds": [8946],
    "use_str": True,
    "use_kmer": True,
    "save_predictions": True,
    "compute_metrics": [ClassificationMetric(ignore_y=-100, average="macro").f1_score],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test_no_labels.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json") else None,
    # "dataset_cls": Dataset,  # For your custom dataset preparation
    "dataset_cls": Dataset,
    "model_cls": OmniGenomeModelForSequenceClassification,
}

bench_config = AutoBenchConfig(config_dict)
