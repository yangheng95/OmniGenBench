# -*- coding: utf-8 -*-
# file: config.py
# time: 23:04 26/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import os

import torch

from omnigenome import (
    ClassificationMetric,
    AutoBenchConfig,
    OmniGenomeModelForTokenClassification,
    OmniGenomeDataset,
)

class OmniGenomeDatasetForTokenClassification(OmniGenomeDataset):
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDatasetForTokenClassification, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        labels = -100
        if isinstance(instance, str):
            sequence = instance
        elif isinstance(instance, dict):
            sequence = (
                instance.get("seq", None)
                if "seq" in instance
                else instance.get("sequence", None)
            )
            label = instance.get("label", None)
            labels = instance.get("labels", None)
            labels = labels if labels is not None else label
            if not sequence:
                raise Exception(
                    "The input instance must contain a 'seq' or 'sequence' key."
                )
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            if len(set(self.label2id.keys()) | set([str(l) for l in labels])) != len(set(self.label2id.keys())):
                print(f"Warning: The labels <{labels}> in the input instance do not match the label2id mapping.")
            labels = (
                [-100]
                + [self.label2id.get(str(l), -100) for l in labels][
                    : self.max_length - 2
                ]
                + [-100]
            )
            tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs


label2id = {"(": 0, ")": 1, ".": 2}

# Hyperparameters
config_dict = {
    "task_name": "RNA-SSP-rnastralign",
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
    "save_predictions": True,
    "compute_metrics": [ClassificationMetric(ignore_y=-100, average="macro").f1_score],
    "train_file": f"{os.path.dirname(__file__)}/train.json",
    "test_file": f"{os.path.dirname(__file__)}/test_no_labels.json",
    "valid_file": f"{os.path.dirname(__file__)}/valid.json"
    if os.path.exists(f"{os.path.dirname(__file__)}/valid.json") else None,
    # "dataset_cls": Dataset,  # For your custom dataset preparation
    "dataset_cls": OmniGenomeDatasetForTokenClassification,
    "model_cls": OmniGenomeModelForTokenClassification,
}

bench_config = AutoBenchConfig(config_dict)
