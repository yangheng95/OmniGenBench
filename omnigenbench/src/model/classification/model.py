# -*- coding: utf-8 -*-
# file: model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import torch

from ...abc.abstract_model import OmniModel
from ..module_utils import OmniPooling


class OmniModelForTokenClassification(OmniModel):
    """
    Model for token classification tasks in genomics.

    This model is designed for token-level classification tasks such as
    sequence labeling, where each token in the input sequence needs to be
    classified into different categories. It extends the base OmniModel
    with token-level classification capabilities.

    The model adds a classification head on top of the base model's hidden
    states and applies softmax to produce probability distributions over
    the label classes for each token.

    Attributes:
        softmax (torch.nn.Softmax): Softmax layer for probability computation.
        classifier (torch.nn.Linear): Linear classification head.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function for training.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the token classification model.

        Args:
            config_or_model: Model configuration, pre-trained model path, or model instance.
            tokenizer: The tokenizer associated with the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Example:
            >>> model = OmniModelForTokenClassification("model_path", tokenizer)
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for token classification.

        This method performs the forward pass through the model, computing
        logits for each token in the input sequence and applying softmax
        to produce probability distributions.

        Args:
            **inputs: Input tensors including 'input_ids', 'attention_mask',
                     and optionally 'labels'.

        Returns:
            dict: A dictionary containing:
                - logits: Token-level classification logits
                - last_hidden_state: Final hidden states from the base model
                - labels: Ground truth labels (if provided)

        Example:
            >>> outputs = model(
            ...     input_ids=torch.tensor([[1, 2, 3, 4]]),
            ...     attention_mask=torch.tensor([[1, 1, 1, 1]]),
            ...     labels=torch.tensor([[0, 1, 0, 1]])
            ... )
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
        Performs token-level prediction on raw inputs.

        This method takes raw sequences or tokenized inputs and returns
        token-level predictions. It processes the inputs through the model
        and returns the predicted class for each token.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Predicted class indices for each token
                - logits: Raw logits from the model
                - last_hidden_state: Final hidden states

        Example:
            >>> # Predict on a single sequence
            >>> outputs = model.predict("ATCGATCG")
            >>> print(outputs['predictions'].shape)  # (seq_len,)

            >>> # Predict on multiple sequences
            >>> outputs = model.predict(["ATCGATCG", "GCTAGCTA"])
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
        Performs token-level inference with human-readable output.

        This method provides processed, human-readable token-level predictions.
        It converts logits to class labels and handles special tokens appropriately.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Human-readable class labels for each token
                - logits: Raw logits from the model
                - confidence: Confidence scores for predictions
                - last_hidden_state: Final hidden states

        Example:
            >>> # Inference on a single sequence
            >>> results = model.inference("ATCGATCG")
            >>> print(results['predictions'])  # ['A', 'T', 'C', 'G', ...]
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
        """
        Calculates the cross-entropy loss for token classification.

        This method computes the cross-entropy loss between the predicted
        logits and the ground truth labels, ignoring padding tokens.

        Args:
            logits (torch.Tensor): Predicted logits from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed loss value.

        Example:
            >>> loss = model.loss_function(logits, labels)
        """
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniModelForSequenceClassification(OmniModel):
    """
    Model for sequence classification tasks in genomics.

    This model is designed for sequence-level classification tasks where
    the entire input sequence is classified into one of several categories.
    It extends the base OmniModel with sequence-level classification capabilities.

    The model uses a pooling mechanism to aggregate token-level representations
    into a sequence-level representation, which is then classified using a
    linear classifier.

    Attributes:
        pooler (OmniPooling): Pooling layer for sequence-level representation.
        softmax (torch.nn.Softmax): Softmax layer for probability computation.
        classifier (torch.nn.Linear): Linear classification head.
        loss_fn (torch.nn.CrossEntropyLoss): Loss function for training.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the sequence classification model.

        Args:
            config_or_model: Model configuration, pre-trained model path, or model instance.
            tokenizer: The tokenizer associated with the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Example:
            >>> model = OmniModelForSequenceClassification("model_path", tokenizer)
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniPooling(self.config)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for sequence classification.

        This method performs the forward pass through the model, computing
        sequence-level logits and applying softmax to produce probability
        distributions over the label classes.

        Args:
            **inputs: Input tensors including 'input_ids', 'attention_mask',
                     and optionally 'labels'.

        Returns:
            dict: A dictionary containing:
                - logits: Sequence-level classification logits
                - last_hidden_state: Final hidden states from the base model
                - labels: Ground truth labels (if provided)

        Example:
            >>> outputs = model(
            ...     input_ids=torch.tensor([[1, 2, 3, 4]]),
            ...     attention_mask=torch.tensor([[1, 1, 1, 1]]),
            ...     labels=torch.tensor([0])
            ... )
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
        Performs sequence-level prediction on raw inputs.

        This method takes raw sequences or tokenized inputs and returns
        sequence-level predictions. It processes the inputs through the model
        and returns the predicted class for each sequence.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Predicted class indices for each sequence
                - logits: Raw logits from the model
                - last_hidden_state: Final hidden states

        Example:
            >>> # Predict on a single sequence
            >>> outputs = model.predict("ATCGATCG")
            >>> print(outputs['predictions'])  # tensor([0])

            >>> # Predict on multiple sequences
            >>> outputs = model.predict(["ATCGATCG", "GCTAGCTA"])
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
        Performs sequence-level inference with human-readable output.

        This method provides processed, human-readable sequence-level predictions.
        It converts logits to class labels and provides confidence scores.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Human-readable class labels for each sequence
                - logits: Raw logits from the model
                - confidence: Confidence scores for predictions
                - last_hidden_state: Final hidden states

        Example:
            >>> # Inference on a single sequence
            >>> results = model.inference("ATCGATCG")
            >>> print(results['predictions'])  # "positive"
            >>> print(results['confidence'])   # 0.95
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
        """
        Calculates the cross-entropy loss for sequence classification.

        This method computes the cross-entropy loss between the predicted
        logits and the ground truth labels.

        Args:
            logits (torch.Tensor): Predicted logits from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed loss value.

        Example:
            >>> loss = model.loss_function(logits, labels)
        """
        loss = self.loss_fn(logits.view(-1, self.config.num_labels), labels.view(-1))
        return loss


class OmniModelForMultiLabelSequenceClassification(OmniModelForSequenceClassification):
    """
    Model for multi-label sequence classification tasks in genomics.

    This model is designed for multi-label classification tasks where
    a single sequence can be assigned multiple labels simultaneously.
    It extends the sequence classification model with multi-label capabilities.

    The model uses sigmoid activation instead of softmax to allow multiple
    labels per sequence and uses binary cross-entropy loss for training.

    Attributes:
        softmax (torch.nn.Sigmoid): Sigmoid layer for multi-label probability computation.
        loss_fn (torch.nn.BCELoss): Binary cross-entropy loss function for training.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initializes the multi-label sequence classification model.

        Args:
            config_or_model: Model configuration, pre-trained model path, or model instance.
            tokenizer: The tokenizer associated with the model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Example:
            >>> model = OmniModelForMultiLabelSequenceClassification("model_path", tokenizer)
        """
        self.threshold = kwargs.pop("threshold", 0.5)
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()

    def loss_function(self, logits, labels):
        """
        Calculates the binary cross-entropy loss for multi-label classification.

        This method computes the binary cross-entropy loss between the predicted
        probabilities and the ground truth multi-label targets.

        Args:
            logits (torch.Tensor): Predicted logits from the model.
            labels (torch.Tensor): Ground truth multi-label targets.

        Returns:
            torch.Tensor: The computed loss value.

        Example:
            >>> loss = model.loss_function(logits, labels)
        """
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Performs multi-label prediction on raw inputs.

        This method takes raw sequences or tokenized inputs and returns
        multi-label predictions. It applies a threshold to determine
        which labels are active for each sequence.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Multi-label predictions for each sequence
                - logits: Raw logits from the model
                - last_hidden_state: Final hidden states

        Example:
            >>> # Predict on a single sequence
            >>> outputs = model.predict("ATCGATCG")
            >>> print(outputs['predictions'])  # tensor([1, 0, 1, 0])
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].cpu()
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
        """
        Performs multi-label inference with human-readable output.

        This method provides processed, human-readable multi-label predictions.
        It converts logits to binary labels and provides confidence scores.

        Args:
            sequence_or_inputs: A sequence (str), list of sequences, or
                               tokenized inputs (dict/tuple).
            **kwargs: Additional arguments for tokenization and inference.

        Returns:
            dict: A dictionary containing:
                - predictions: Human-readable binary labels for each sequence
                - logits: Raw logits from the model
                - confidence: Confidence scores for predictions
                - last_hidden_state: Final hidden states

        Example:
            >>> # Inference on a single sequence
            >>> results = model.inference("ATCGATCG")
            >>> print(results['predictions'])  # tensor([1, 0, 1, 0])
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(self.threshold).to(torch.int).cpu()
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


class OmniModelForTokenClassificationWith2DStructure(OmniModelForTokenClassification):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniPooling(self.config)
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


class OmniModelForSequenceClassificationWith2DStructure(
    OmniModelForSequenceClassification
):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniPooling(self.config)
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


class OmniModelForMultiLabelSequenceClassificationWith2DStructure(
    OmniModelForSequenceClassificationWith2DStructure
):
    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        self.threshold = kwargs.pop("threshold", 0.5)
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.softmax = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.model_info()

    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].cpu()
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
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            prediction = logits[i].ge(self.threshold).to(torch.int).cpu()
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
        loss = self.loss_fn(logits.view(-1), labels.view(-1).to(torch.float32))
        return loss
