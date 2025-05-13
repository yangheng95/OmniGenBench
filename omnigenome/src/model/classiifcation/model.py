# -*- coding: utf-8 -*-
# file: model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch

from ...abc.abstract_model import OmniGenomeModel
from ..module_utils import OmniGenomePooling


class OmniGenomeModelForTokenClassification(OmniGenomeModel):
    """ 
    This class defines a model for token classification, extending from OmniGenomeModel.
    It includes the necessary components such as a classifier, softmax layer, and cross-entropy loss function.
    """
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        """
        Initializes the OmniGenomeModelForTokenClassification with a configuration, tokenizer, and other parameters.
        Sets up the classifier and loss function.
        """
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for the model. Computes the logits for token classification.
        Applies dropout, activation, and the classifier to the hidden state.
        """
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Makes predictions based on input sequences. 
        Extracts logits and the last hidden state, then returns predicted labels and other relevant outputs.
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].argmax(dim=-1).detach().cpu())

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
        """
        Performs inference on the input data and returns predictions and additional outputs.
        This method provides confidence scores along with the predicted labels.
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        inputs = raw_outputs["inputs"]
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            # Note that the first and last tokens are removed,
            # and the length of outputs are calculated based on the tokenized inputs.
            i_logit = logits[i][inputs["input_ids"][i].ne(self.config.pad_token_id)][
                1:-1
            ]
            prediction = [
                self.config.id2label.get(x.item(), "") for x in i_logit.argmax(dim=-1)
            ]
            predictions.append(prediction)

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "confidence": torch.max(logits[0]),
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "confidence": torch.max(logits, dim=-1)[0],
                "last_hidden_state": last_hidden_state,
            }

        return outputs

    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniGenomeModelForSequenceClassification(OmniGenomeModel):
    """ 
    This class defines a model for sequence classification, with pooling and softmax layers, 
    as well as a cross-entropy loss function for training.
    """
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the OmniGenomeModelForSequenceClassification with a configuration, tokenizer, and other parameters.
        Sets up the pooler, classifier, and loss function.
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for the sequence classification model. 
        Computes logits for the entire sequence using the last hidden state, pooling, and the classifier.
        """
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Makes predictions for sequence classification. 
        Outputs the predicted labels, logits, and last hidden state.
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].argmax(dim=-1))

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
        """
        Performs inference on the input sequences for sequence classification.
        Provides predicted labels along with logits and confidence scores.
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(
                self.config.id2label.get(logits[i].argmax(dim=-1).item(), "")
            )

        if not isinstance(sequence_or_inputs, list):
            outputs = {
                "predictions": predictions[0],
                "logits": logits[0],
                "confidence": torch.max(logits[0]),
                "last_hidden_state": last_hidden_state[0],
            }
        else:
            outputs = {
                "predictions": predictions,
                "logits": logits,
                "confidence": torch.max(logits, dim=-1)[0],
                "last_hidden_state": last_hidden_state,
            }

        return outputs

    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniGenomeModelForMultiLabelSequenceClassification(
    OmniGenomeModelForSequenceClassification
):
    """ 
    This class is an extension of OmniGenomeModelForSequenceClassification for multi-label sequence classification tasks. 
    It uses a sigmoid activation and binary cross-entropy loss.
    """
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the OmniGenomeModelForMultiLabelSequenceClassification with the required components,
        including the sigmoid activation and binary cross-entropy loss.
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()

    def loss_function(self, logits, labels):
        """
        Computes the loss using binary cross-entropy for multi-label classification.
        """
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Predicts multi-label classification outcomes based on logits.
        Uses a threshold of 0.5 for label assignment.
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(0.5).to(torch.int).cpu()
            predictions.append(prediction)

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
        return self.predict(sequence_or_inputs, **kwargs)


class OmniGenomeModelForTokenClassificationWith2DStructure(
    OmniGenomeModelForTokenClassification
):
    """ 
    This class extends OmniGenomeModelForTokenClassification and adds support for 2D structures 
    in token classification tasks. It utilizes a pooling layer to process the input sequence 
    and outputs the logits, last hidden states, and labels.
    """
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.model_info()

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs


class OmniGenomeModelForSequenceClassificationWith2DStructure(
    OmniGenomeModelForSequenceClassification
):
    """
    This class extends OmniGenomeModelForSequenceClassification and adds support for 2D structures
    in sequence classification tasks. The pooling layer is applied to the hidden states before classification.
    """
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniGenomePooling(self.config)
        self.model_info()

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        logits = self.softmax(logits)

        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs


class OmniGenomeModelForMultiLabelSequenceClassificationWith2DStructure(
    OmniGenomeModelForSequenceClassificationWith2DStructure
):
    """
    This class extends OmniGenomeModelForSequenceClassificationWith2DStructure and supports 
    multi-label classification tasks. The model uses a sigmoid activation function for multi-label classification 
    and computes the binary cross-entropy loss.
    """
    def __init__(self, config_or_model_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()

    def loss_function(self, logits, labels):
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss

    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(0.5).to(torch.int).cpu()
            predictions.append(prediction)

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
        return self.predict(sequence_or_inputs, **kwargs)
