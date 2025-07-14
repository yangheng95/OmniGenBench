# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Specialized dataset classes for OmniGenome framework.

This module provides specialized dataset classes for various genomic tasks,
inheriting from the abstract `OmniDataset`. These classes handle data preparation
for token classification, sequence classification, token regression, and sequence regression,
integrating with tokenizers and managing metadata.
"""
import json

import numpy as np
import torch

from ..abc.abstract_dataset import OmniDataset
from ..misc.utils import fprint
from ... import __name__, __version__


class OmniDatasetForTokenClassification(OmniDataset):
    """
    Dataset class specifically designed for token classification tasks in genomics.

    This class extends `OmniDataset` to provide functionalities for preparing input sequences
    and their corresponding token-level labels. It's designed for tasks where each token
    in a sequence needs to be classified independently.

    Attributes:
        metadata: Dictionary containing dataset metadata including library information
        label2id: Mapping from label strings to integer IDs
    """

    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for token classification.

        Args:
            data_source: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForTokenClassification, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )
        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_token_classification",
            }
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare a single data instance for token classification.

        This method handles both string sequences and dictionary instances
        containing sequence and label information. It tokenizes the input
        sequence and prepares token-level labels for classification.

        Args:
            instance: A single data instance. Can be a string representing the sequence
                     or a dictionary with 'seq'/'sequence' and 'labels'/'label' keys.
            **kwargs: Additional keyword arguments for tokenization, such as 'padding'
                     and 'truncation'.

        Returns:
            dict: A dictionary of tokenized inputs, including 'input_ids', 'attention_mask',
                  and 'labels' (tensor of token-level labels).

        Raises:
            Exception: If the input instance format is unknown or if a dictionary
                      instance does not contain a 'seq' or 'sequence' key.
        """
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
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            if len(set(self.label2id.keys()) | set([str(l) for l in labels])) != len(
                set(self.label2id.keys())
            ):
                fprint(
                    f"Warning: The labels <{labels}> in the input instance do not match the label2id mapping."
                )
            labels = (
                [-100]
                + [self.label2id.get(str(l), -100) for l in labels][
                    : self.max_length - 2
                ]
                + [-100]
            )

        tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs


class OmniDatasetForSequenceClassification(OmniDataset):
    """
    Dataset class for sequence classification tasks in genomics.

    This class extends `OmniDataset` to prepare input sequences and their corresponding
    sequence-level labels. It's designed for tasks where the entire sequence needs
    to be classified into one of several categories.

    Attributes:
        metadata: Dictionary containing dataset metadata including library information
        label2id: Mapping from label strings to integer IDs
    """

    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for sequence classification.

        Args:
            data_source: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForSequenceClassification, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_sequence_classification",
            }
        )
        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare a single data instance for sequence classification.

        This method handles both string sequences and dictionary instances
        containing sequence and label information. It tokenizes the input
        sequence and prepares sequence-level labels for classification.

        Args:
            instance: A single data instance. Can be a string representing the sequence
                     or a dictionary with 'seq'/'sequence' and 'labels'/'label' keys.
            **kwargs: Additional keyword arguments for tokenization, such as 'padding'
                     and 'truncation'.

        Returns:
            dict: A dictionary of tokenized inputs, including 'input_ids', 'attention_mask',
                  and 'labels' (tensor of sequence-level labels).

        Raises:
            Exception: If the input instance format is unknown or if a dictionary
                      instance does not contain a 'label' or 'labels' key, or if
                      the label is not an integer.
        """
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
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            if not isinstance(labels, int):
                raise Exception(
                    "The label must be an integer for sequence classification."
                )
            labels = self.label2id.get(str(labels), -100)

        tokenized_inputs["labels"] = torch.tensor(labels)
        return tokenized_inputs


class OmniDatasetForTokenRegression(OmniDataset):
    """
    Dataset class for token regression tasks in genomics.

    This class extends `OmniDataset` to prepare input sequences and their corresponding
    token-level regression targets. It's designed for tasks where each token in a
    sequence needs to be assigned a continuous value.

    Attributes:
        metadata: Dictionary containing dataset metadata including library information
    """

    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for token regression.

        Args:
            data_source: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForTokenRegression, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_token_regression",
            }
        )
        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare a single data instance for token regression.

        This method handles both string sequences and dictionary instances
        containing sequence and regression target information. It tokenizes
        the input sequence and prepares token-level regression targets.

        Args:
            instance: A single data instance. Can be a string representing the sequence
                     or a dictionary with 'seq'/'sequence' and 'labels'/'label' keys.
            **kwargs: Additional keyword arguments for tokenization, such as 'padding'
                     and 'truncation'.

        Returns:
            dict: A dictionary of tokenized inputs, including 'input_ids', 'attention_mask',
                  and 'labels' (tensor of token-level regression targets).

        Raises:
            Exception: If the input instance format is unknown or if a dictionary
                      instance does not contain a 'seq' or 'sequence' key.
        """
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
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            # Handle token-level regression labels
            if isinstance(labels, (list, tuple)):
                # Ensure labels match sequence length
                labels = list(labels)[
                    : self.max_length - 2
                ]  # Account for special tokens
                labels = [-100] + labels + [-100]  # Add padding for special tokens
            else:
                # Single value for the entire sequence
                labels = [-100] + [float(labels)] * (self.max_length - 2) + [-100]

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        return tokenized_inputs


class OmniDatasetForSequenceRegression(OmniDataset):
    """
    Dataset class for sequence regression tasks in genomics.

    This class extends `OmniDataset` to prepare input sequences and their corresponding
    sequence-level regression targets. It's designed for tasks where the entire
    sequence needs to be assigned a continuous value.

    Attributes:
        metadata: Dictionary containing dataset metadata including library information
    """

    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for sequence regression.

        Args:
            data_source: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForSequenceRegression, self).__init__(
            data_source, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": __name__,
                "omnigenome_version": __version__,
                "task": "genome_sequence_regression",
            }
        )
        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare a single data instance for sequence regression.

        This method handles both string sequences and dictionary instances
        containing sequence and regression target information. It tokenizes
        the input sequence and prepares sequence-level regression targets.

        Args:
            instance: A single data instance. Can be a string representing the sequence
                     or a dictionary with 'seq'/'sequence' and 'labels'/'label' keys.
            **kwargs: Additional keyword arguments for tokenization, such as 'padding'
                     and 'truncation'.

        Returns:
            dict: A dictionary of tokenized inputs, including 'input_ids', 'attention_mask',
                  and 'labels' (tensor of sequence-level regression targets).

        Raises:
            Exception: If the input instance format is unknown or if a dictionary
                      instance does not contain a 'label' or 'labels' key.
        """
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
        else:
            raise Exception("Unknown instance format.")

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )
        for col in tokenized_inputs:
            tokenized_inputs[col] = tokenized_inputs[col].squeeze()

        if labels is not None:
            # Convert to float for regression
            labels = float(labels)

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
        return tokenized_inputs
