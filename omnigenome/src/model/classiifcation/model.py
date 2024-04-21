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


class OmniGenomeEncoderModelForTokenClassification(OmniGenomeModel):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeEncoderModelForTokenClassification"
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state = extract_last_hidden_state(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
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
            predictions.append(logits[i].argmax(dim=-1).detach().cpu().numpy())

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
            i_logit = logits[i][
                : inputs["input_ids"][i].ne(self.config.pad_token_id).sum(dim=-1)
            ]
            prediction = [
                self.config.id2label.get(x.item(), "") for x in i_logit.argmax(dim=-1)
            ]
            predictions.append(prediction)

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
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniGenomeEncoderModelForSequenceClassification(OmniGenomeModel):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeEncoderModelForSequenceClassification"
        self.pooler = BertPooler(config)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        last_hidden_state = extract_last_hidden_state(self.model, inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        pooled_output = self.pooler(last_hidden_state)
        logits = self.classifier(pooled_output)
        logits = self.softmax(logits)
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
            predictions.append(logits[i].argmax(dim=-1).item())

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
            predictions.append(
                self.config.id2label.get(logits[i].argmax(dim=-1).item(), "")
            )

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
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniGenomeEncoderModelForTokenClassificationWith2DStructure(
    OmniGenomeEncoderModelForSequenceClassification
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeEncoderModelForTokenClassificationWith2DStructure"

        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

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
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs


class OmniGenomeEncoderModelForSequenceClassificationWith2DStructure(
    OmniGenomeEncoderModelForSequenceClassification
):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata[
            "model_name"
        ] = "OmniGenomeEncoderModelForSequenceClassificationWith2DStructure"

        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

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
        last_hidden_state = self.pooler(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs


class OmniGenomeDecoderModelForTokenClassification(
    OmniGenomeEncoderModelForTokenClassification
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
        self.metadata["model_name"] = "OmniGenomeDecoderModelForTokenClassification"

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state = extract_last_hidden_state(self.model, (input_ids, None))
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {"logits": logits, "last_hidden_state": last_hidden_state}
        return outputs


class OmniGenomeDecoderModelForSequenceClassification(
    OmniGenomeEncoderModelForSequenceClassification
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
        self.metadata["model_name"] = "OmniGenomeDecoderModelForSequenceClassification"

        self.pooler = BertPooler(config)

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        last_hidden_state = extract_last_hidden_state(self.model, (input_ids, None))
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1) - 1
        pooled_logits = logits[
            torch.arange(input_ids.size(0), device=logits.device), sequence_lengths
        ]
        outputs = {"logits": pooled_logits, "last_hidden_state": last_hidden_state}
        return outputs


class OmniGenomeDecoderModelForTokenClassificationWith2DStructure(
    OmniGenomeEncoderModelForTokenClassification
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
        ] = "OmniGenomeDecoderModelForTokenClassificationWith2DStructure"

        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pooler = BertPooler(config)

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
        logits = self.softmax(logits)

        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs


class OmniGenomeDecoderModelForSequenceClassificationWith2DStructure(
    OmniGenomeEncoderModelForSequenceClassification
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
        ] = "OmniGenomeDecoderModelForSequenceClassificationWith2DStructure"

        self.cat_layer = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.conv1d = torch.nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.pooler = BertPooler(config)

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
        logits = self.softmax(logits)
        sequence_lengths = input_ids.ne(self.config.pad_token_id).sum(dim=-1) - 1
        pooled_logits = logits[
            torch.arange(input_ids.size(0), device=logits.device), sequence_lengths
        ]
        outputs = {
            "logits": pooled_logits,
            "last_hidden_state": last_hidden_state,
            "ss_last_hidden_state": ss_last_hidden_state,
        }
        return outputs
