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
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
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


def mcrmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = y_true != -100
    filtered_y_pred = y_pred[mask]
    filtered_y_true = y_true[mask]
    rmse_per_target = np.sqrt(np.mean((filtered_y_true - filtered_y_pred) ** 2, axis=0))
    mcrmse_value = np.mean(rmse_per_target)
    return mcrmse_value


class Model(OmniModelForTokenRegression):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.MSELoss()
        # self.model_info()

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].cpu())

        outputs = {
            "predictions": (
                torch.vstack(predictions).to(self.model.device)
                if predictions[0].shape
                else torch.tensor(predictions).to(self.model.device)
            ),
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        inputs = raw_outputs["inputs"]
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            i_logit = logits[i][inputs["input_ids"][i].ne(self.config.pad_token_id)][
                      1:-1
                      ]
            predictions.append(i_logit.detach().cpu())

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "last_hidden_state": last_hidden_state,
            }

        return outputs

    def loss_function(self, logits, labels):
        padding_value = (
            self.config.ignore_y if hasattr(self.config, "ignore_y") else -100
        )
        logits = logits.view(-1)
        labels = labels.view(-1)
        mask = torch.where(labels != padding_value)

        filtered_logits = logits[mask]
        filtered_targets = labels[mask]

        loss = self.loss_fn(filtered_logits, filtered_targets)
        return loss


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
