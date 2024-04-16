# -*- coding: utf-8 -*-
# file: model.py
# time: 13:30 10/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch
from transformers import BatchEncoding

from ...abc.abstract_model import OmniGenomeModel


class OmniGenomeEncoderModelForMLM(OmniGenomeModel):
    def __init__(self, config, base_model, tokenizer, *args, **kwargs):
        super().__init__(config, base_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = "OmniGenomeEncoderModelForMLM"
        if not hasattr(base_model, "lm_head"):
            raise ValueError(
                "The model does not have a language model head, which is required for MLM."
                "Please use a model that supports masked language modeling."
            )
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = (
            outputs["last_hidden_state"]
            if "last_hidden_state" in outputs
            else outputs["hidden_states"][-1]
        )
        logits = outputs["logits"] if "logits" in outputs else None
        loss = outputs["loss"] if "loss" in outputs else None
        outputs = {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }
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

    def inference(self, sequence_or_inputs, **kwargs):
        inputs = self.tokenizer(sequence_or_inputs, return_tensors="pt", **kwargs)
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self(inputs)
        logits = outputs["logits"]
        last_hidden_state = outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            i_logits = logits[i][
                : inputs["input_ids"][i].ne(self.tokenizer.pad_token_id).sum().item()
            ][1:-1]
            prediction = self.tokenizer.decode(i_logits.argmax(dim=-1))
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
