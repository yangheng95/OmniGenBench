# -*- coding: utf-8 -*-
# file: model.py
# time: 18:36 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
Regression models for OmniGenome framework.

This module provides various regression model implementations for genomic sequence analysis,
including token-level regression, sequence-level regression, structural imputation,
and matrix regression/classification tasks.
"""
import torch

from .resnet import resnet_b16
from ...abc.abstract_model import OmniModel
from ..module_utils import OmniPooling


class OmniModelForTokenRegression(OmniModel):
    """
    Token-level regression model for genomic sequences.

    This model performs regression at the token level, predicting continuous values
    for each token in the input sequence. It's useful for tasks like predicting
    binding affinities, expression levels, or other continuous properties at each
    position in a genomic sequence.

    Attributes:
        classifier: Linear layer for regression output
        loss_fn: Mean squared error loss function
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the token regression model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.MSELoss()
        # self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for token-level regression.

        Args:
            **inputs: Input tensors including input_ids, attention_mask, and labels

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
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
        """
        Generate predictions for token-level regression.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
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
        """
        Perform inference for token-level regression, excluding special tokens.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
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
        """
        Compute the loss for token-level regression.

        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
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


class OmniModelForSequenceRegression(OmniModel):
    """
    This model performs regression at the sequence level, predicting a single
    continuous value for the entire input sequence. It's useful for tasks like
    predicting overall expression levels, binding affinities, or other sequence-level
    properties.

    Attributes:
        pooler: OmniPooling layer for sequence-level representation
        classifier: Linear layer for regression output
        loss_fn: Mean squared error loss function
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the sequence regression model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(
            self.config.hidden_size, self.config.num_labels
        )
        self.loss_fn = torch.nn.MSELoss()
        # self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for sequence-level regression.

        Args:
            **inputs: Input tensors including input_ids, attention_mask, and labels

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Generate predictions for sequence-level regression.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
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
        """
        Perform inference for sequence-level regression.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].cpu())

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
        """
        Compute the loss for sequence-level regression.

        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
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


class OmniModelForStructuralImputation(OmniModelForSequenceRegression):
    """
    This model is specialized for imputing missing structural information in
    genomic sequences. It extends the sequence regression model with additional
    embedding capabilities for structural features.

    Attributes:
        embedding: Embedding layer for structural features
        loss_fn: Mean squared error loss function
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the structural imputation model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.loss_fn = torch.nn.MSELoss()
        self.embedding = torch.nn.Embedding(1, self.config.hidden_size)
        # self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for structural imputation.

        Args:
            **inputs: Input tensors including input_ids, attention_mask, and labels

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs


class OmniModelForTokenRegressionWith2DStructure(OmniModelForTokenRegression):
    """
    This model extends the basic token regression model to incorporate
    2D structural information, useful for RNA structure prediction
    and other structural genomics tasks.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the 2D structure-aware token regression model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__

    def forward(self, **inputs):
        """
        Forward pass for 2D structure-aware token regression.

        Args:
            **inputs: Input tensors including input_ids, attention_mask, labels, and structural info

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
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


class OmniModelForSequenceRegressionWith2DStructure(OmniModelForSequenceRegression):
    """
    This model extends the basic sequence regression model to incorporate
    2D structural information, useful for RNA structure prediction
    and other structural genomics tasks.
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the 2D structure-aware sequence regression model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__

    def forward(self, **inputs):
        """
        Forward pass for 2D structure-aware sequence regression.

        Args:
            **inputs: Input tensors including input_ids, attention_mask, labels, and structural info

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
        labels = inputs.pop("labels", None)
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        logits = self.classifier(last_hidden_state)
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels,
        }
        return outputs


class OmniModelForMatrixRegression(OmniModel):
    """
    This model performs regression on matrix representations of genomic sequences,
    useful for tasks like contact map prediction, structure prediction, or other
    matrix-based genomic analysis tasks.

    Attributes:
        resnet: ResNet backbone for processing matrix inputs
        classifier: Linear layer for regression output
        loss_fn: Mean squared error loss function
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the matrix regression model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.resnet = resnet_b16(channels=128, bbn=16)
        self.classifier = torch.nn.Linear(1, self.config.num_labels)
        self.loss_fn = torch.nn.MSELoss()
        # self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for matrix regression.

        Args:
            **inputs: Input tensors including matrix representations and labels

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
        labels = inputs.pop("labels", None)
        matrix_inputs = inputs.pop("matrix_inputs", None)

        if matrix_inputs is None:
            raise ValueError("matrix_inputs is required for matrix regression")

        outputs = self.resnet(matrix_inputs)
        logits = self.classifier(outputs)

        outputs = {
            "logits": logits,
            "last_hidden_state": outputs,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Generate predictions for matrix regression.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
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
        """
        Perform inference for matrix regression.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            predictions.append(logits[i].cpu())

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
        """
        Compute the loss for matrix regression.

        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
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


class OmniModelForMatrixClassification(OmniModel):
    """
    This model performs classification on matrix representations of genomic sequences,
    useful for tasks like structure classification, contact map classification, or other
    matrix-based genomic analysis tasks.

    Attributes:
        resnet: ResNet backbone for processing matrix inputs
        classifier: Linear layer for classification output
        loss_fn: Cross-entropy loss function
    """

    def __init__(self, config_or_model, tokenizer, *args, **kwargs):
        """
        Initialize the matrix classification model.

        Args:
            config_or_model: Model configuration or pre-trained model
            tokenizer: Tokenizer for processing input sequences
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_or_model, tokenizer, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        # For binary classification, output size is 1
        self.classifier = torch.nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # Change to BCEWithLogitsLoss for binary classification
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.cnn = resnet_b16(channels=self.config.hidden_size, bbn=16)
        # self.model_info()

    def forward(self, **inputs):
        """
        Forward pass for matrix classification.

        Args:
            **inputs: Input tensors including matrix representations and labels

        Returns:
            dict: Dictionary containing logits, last_hidden_state, and labels
        """
        labels = inputs.pop("labels", None)
        matrix_inputs = inputs.pop("matrix_inputs", None)

        if matrix_inputs is None:
            raise ValueError("matrix_inputs is required for matrix classification")

        outputs = self.resnet(matrix_inputs)
        logits = self.classifier(outputs)

        outputs = {
            "logits": logits,
            "last_hidden_state": outputs,
            "labels": labels,
        }
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """
        Generate predictions for matrix classification.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        for i in range(logits.shape[0]):
            # Apply sigmoid for binary classification
            pred_class = (logits[i] > 0.5).float()
            predictions.append(pred_class.cpu())
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
        Perform inference for matrix classification.

        Args:
            sequence_or_inputs: Input sequences or pre-processed inputs
            **kwargs: Additional keyword arguments

        Returns:
            dict: Dictionary containing predictions, logits, and last_hidden_state
        """
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)
        inputs = raw_outputs["inputs"]
        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        predictions = []
        probabilities = []
        for i in range(logits.shape[0]):
            i_logit = logits[i][inputs["input_ids"][i].ne(self.config.pad_token_id)][
                1:-1
            ]
            probs = i_logit
            # For binary classification, threshold at 0.5
            pred_class = (probs > 0.5).float()
            predictions.append(pred_class.detach().cpu())
            probabilities.append(probs.detach().cpu())

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
        """
        Compute the loss for matrix classification.

        Args:
            logits (torch.Tensor): Model predictions
            labels (torch.Tensor): Ground truth labels

        Returns:
            torch.Tensor: Computed loss value
        """
        padding_value = (
            self.config.ignore_y if hasattr(self.config, "ignore_y") else -100
        )
        logits = logits.view(-1, self.config.num_labels)
        labels = labels.view(-1)
        mask = torch.where(labels != padding_value)

        # Filter out padding
        filtered_logits = logits[mask]
        filtered_targets = labels[mask]

        # Reshape for binary classification
        filtered_logits = filtered_logits.view(-1)
        filtered_targets = filtered_targets.view(
            -1
        ).float()  # Convert to float for BCEWithLogitsLoss

        # Apply BCEWithLogitsLoss
        loss = self.loss_fn(filtered_logits, filtered_targets)
        return loss
