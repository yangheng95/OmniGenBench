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
from ...abc.abstract_model import extract_last_hidden_state


class OmniGenomeEncoderModelForTokenRegression(OmniGenomeModel):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeEncoderModelForTokenRegression"

        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state = extract_last_hidden_state(self.model, inputs)
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

        loss = torch.nn.functional.mse_loss(
            filtered_logits, filtered_targets, reduction="mean"
        )
        return loss**0.5


class OmniGenomeEncoderModelForSequenceRegression(OmniGenomeModel):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeEncoderModelForSequenceRegression"
        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state = extract_last_hidden_state(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        pooled_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooled_output)
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

        loss = torch.nn.functional.mse_loss(
            filtered_logits, filtered_targets, reduction="mean"
        )
        return loss**0.5


class OmniGenomeEncoderModelForTokenRegressionWith2DStructure(
    OmniGenomeEncoderModelForSequenceRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeEncoderModelForTokenRegressionWith2DStructure"
        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state, ss_last_hidden_state = extract_last_hidden_state(
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


class OmniGenomeEncoderModelForSequenceRegressionWith2DStructure(
    OmniGenomeEncoderModelForSequenceRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeEncoderModelForSequenceRegressionWith2DStructure"
        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state, ss_last_hidden_state = extract_last_hidden_state(
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


class OmniGenomeDecoderModelForTokenRegression(
    OmniGenomeEncoderModelForTokenRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        if hasattr(config, "n_embd"):
            config.hidden_size = config.n_embd
        elif hasattr(config, "d_model"):
            config.hidden_size = config.d_model
        elif hasattr(config, "hidden_size"):
            config.hidden_size = config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeDecoderModelForTokenRegression"
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state = extract_last_hidden_state(self.model, (input_ids, None))
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1) - 1
        pooled_logits = logits[
            torch.arange(input_ids.size(0), device=logits.device), sequence_lengths
        ]
        outputs = {"logits": pooled_logits, "last_hidden_state": last_hidden_state}
        return outputs


class OmniGenomeDecoderModelForSequenceRegression(
    OmniGenomeEncoderModelForSequenceRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        if hasattr(config, "n_embd"):
            config.hidden_size = config.n_embd
        elif hasattr(config, "d_model"):
            config.hidden_size = config.d_model
        elif hasattr(config, "hidden_size"):
            config.hidden_size = config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeDecoderModelForSequenceRegression"
        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state = extract_last_hidden_state(self.model, (input_ids, None))
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {"logits": logits, "last_hidden_state": last_hidden_state}
        return outputs


class OmniGenomeDecoderModelForTokenRegressionWith2DStructure(
    OmniGenomeEncoderModelForTokenRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        if hasattr(config, "n_embd"):
            config.hidden_size = config.n_embd
        elif hasattr(config, "d_model"):
            config.hidden_size = config.d_model
        elif hasattr(config, "hidden_size"):
            config.hidden_size = config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeDecoderModelForTokenRegressionWith2DStructure"
        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state, ss_last_hidden_state = extract_last_hidden_state(
            self.model, (input_ids, None), ss="viennarna", tokenizer=self.tokenizer
        )

        cat_last_hidden_state = torch.cat(
            [last_hidden_state, ss_last_hidden_state], dim=-1
        )
        conv_output = self.conv1d(cat_last_hidden_state.transpose(1, 2)).transpose(1, 2)
        last_hidden_state = self.cat_layer(
            torch.cat([last_hidden_state, conv_output], dim=-1)
        )

        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1) - 1
        last_hidden_state = last_hidden_state[
            torch.arange(input_ids.size(0), device=last_hidden_state.device),
            sequence_lengths,
        ]
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.activation(logits)

        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs


class OmniGenomeDecoderModelForSequenceRegressionWith2DStructure(
    OmniGenomeEncoderModelForSequenceRegression
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        if hasattr(config, "n_embd"):
            config.hidden_size = config.n_embd
        elif hasattr(config, "d_model"):
            config.hidden_size = config.d_model
        elif hasattr(config, "hidden_size"):
            config.hidden_size = config.hidden_size
        else:
            raise RuntimeError(
                "The hidden size of the model is not found in the config."
            )

        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeDecoderModelForSequenceRegressionWith2DStructure"
        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pooler = BertPooler(config)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state, ss_last_hidden_state = extract_last_hidden_state(
            self.model, (input_ids, None), ss="viennarna", tokenizer=self.tokenizer
        )

        cat_last_hidden_state = torch.cat(
            [last_hidden_state, ss_last_hidden_state], dim=-1
        )
        conv_output = self.conv1d(cat_last_hidden_state.transpose(1, 2)).transpose(1, 2)

        last_hidden_state = self.cat_layer(
            torch.cat([last_hidden_state, conv_output], dim=-1)
        )

        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1) - 1
        last_hidden_state = last_hidden_state[
            torch.arange(input_ids.size(0), device=last_hidden_state.device),
            sequence_lengths,
        ]
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.activation(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs
