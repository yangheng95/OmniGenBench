# -*- coding: utf-8 -*-
# file: model.py
# time: 13:30 10/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import numpy as np
import torch
from transformers import BatchEncoding

from ...abc.abstract_model import OmniGenomeModel


class OmniGenomeModelForMLM(OmniGenomeModel):
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        if "MaskedLM" not in self.model.__class__.__name__:
            raise ValueError(
                "The model does not have a language model head, which is required for MLM."
                "Please use a model that supports masked language modeling."
            )

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, **inputs):
        inputs = inputs.pop("inputs")
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
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].argmax(dim=-1).cpu())

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": (
                    torch.stack(predictions)
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
            prediction = self.tokenizer.decode(i_logit.argmax(dim=-1)).replace(" ", "")
            predictions.append(list(prediction))

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
        loss = loss_fn(logits.view(-1, self.tokenizer.vocab_size), labels.view(-1))
        return loss
