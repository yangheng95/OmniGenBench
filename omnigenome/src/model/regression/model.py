# -*- coding: utf-8 -*-
# file: model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch
from transformers import BatchEncoding
from transformers.models.bert.modeling_bert import BertPooler

from ...abc.abstract_model import OmniGenomeModel
from ...abc.abstract_model import last_hidden_state_forward


class OmniGenomeModelForTokenRegression(OmniGenomeModel):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__

        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, inputs):
        last_hidden_state = last_hidden_state_forward(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {"logits": logits, "last_hidden_state": last_hidden_state}
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        if not isinstance(sequence_or_inputs, BatchEncoding) and not isinstance(
            sequence_or_inputs, dict
        ):
            inputs = self.tokenizer(sequence_or_inputs, return_tensors="pt", **kwargs)
        else:
            inputs = sequence_or_inputs
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self(inputs)
        logits = outputs["logits"]
        last_hidden_state = outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].detach().cpu().numpy())

        outputs = {
            "predictions": predictions,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        inputs = self.tokenizer(sequence_or_inputs, return_tensors="pt", **kwargs)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self(inputs)
        logits = outputs["logits"][:, 1:-1:, :]
        last_hidden_state = outputs["last_hidden_state"][:, 1:-1:, :]

        predictions = []
        for i in range(logits.shape[0]):
            i_logits = logits[i][
                : inputs["input_ids"][i].ne(self.config.pad_token_id).sum().item()
            ]
            predictions.append(i_logits.detach().cpu().numpy())

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


class OmniGenomeModelForSequenceRegression(OmniGenomeModel):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = BertPooler(self.config)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, inputs):
        last_hidden_state = last_hidden_state_forward(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)

        if self._is_causal_lm():
            logits = self.classifier(last_hidden_state)
            pad_token_id = getattr(self.config, "pad_token_id", -100)
            sequence_lengths = inputs["input_ids"].ne(pad_token_id).sum(dim=1) - 1
            logits = logits[
                torch.arange(inputs["input_ids"].size(0), device=logits.device),
                sequence_lengths,
            ]
        else:
            last_hidden_state = self.pooler(last_hidden_state)
            logits = self.classifier(last_hidden_state)

        outputs = {"logits": logits, "last_hidden_state": last_hidden_state}
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        if not isinstance(sequence_or_inputs, BatchEncoding) and not isinstance(
            sequence_or_inputs, dict
        ):
            inputs = self.tokenizer(sequence_or_inputs, return_tensors="pt", **kwargs)
        else:
            inputs = sequence_or_inputs
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self(inputs)
        logits = outputs["logits"]
        last_hidden_state = outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i][0].item())

        outputs = {
            "predictions": predictions,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        inputs = self.tokenizer(sequence_or_inputs, return_tensors="pt", **kwargs)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self(inputs)
        logits = outputs["logits"]
        last_hidden_state = outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i][0].item())

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


class OmniGenomeModelForTokenRegressionWith2DStructure(
    OmniGenomeModelForSequenceRegression
):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__

        self.cat_layer = torch.nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.config.hidden_size * 2,
            out_channels=self.config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )

    def forward(self, inputs):
        last_hidden_state, ss_last_hidden_state = last_hidden_state_forward(
            self.model, inputs, ss="viennarna", tokenizer=self.tokenizer
        )

        cat_last_hidden_state = torch.cat(
            [last_hidden_state, ss_last_hidden_state], dim=-1
        )
        conv_output = self.conv1d(cat_last_hidden_state.transpose(1, 2)).transpose(1, 2)
        last_hidden_state = self.cat_layer(
            torch.cat([last_hidden_state, conv_output], dim=-1)
        )
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs


class OmniGenomeModelForSequenceRegressionWith2DStructure(
    OmniGenomeModelForSequenceRegression
):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__

        self.cat_layer = torch.nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size
        )
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.config.hidden_size * 2,
            out_channels=self.config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )

    def forward(self, inputs):
        last_hidden_state, ss_last_hidden_state = last_hidden_state_forward(
            self.model, inputs, ss="viennarna", tokenizer=self.tokenizer
        )

        cat_last_hidden_state = torch.cat(
            [last_hidden_state, ss_last_hidden_state], dim=-1
        )
        conv_output = self.conv1d(cat_last_hidden_state.transpose(1, 2)).transpose(1, 2)

        last_hidden_state = self.cat_layer(
            torch.cat([last_hidden_state, conv_output], dim=-1)
        )
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs
