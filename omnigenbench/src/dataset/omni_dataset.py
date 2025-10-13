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
from omnigenbench import __name__ as name
from omnigenbench import __version__ as version


class OmniDatasetForTokenClassification(OmniDataset):
    """
    Dataset class for token-level classification tasks in genomics.

    This class extends `OmniDataset` to support tokenizing genomic sequences
    and aligning them with token-level labels for tasks like sequence tagging.

    Attributes:
        metadata (dict): Dataset metadata including library name, version, and task type.
        label2id (dict): Mapping from label strings to integer IDs used for training.
    """

    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        """
        Initializes the token classification dataset.

        Args:
            dataset_name_or_path (str or list): Path(s) to the dataset file(s). Formats supported
                depend on the base `OmniDataset` class.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to process input sequences.
            max_length (int, optional): Maximum sequence length for tokenization. Sequences longer
                than this will be truncated. If None, uses the tokenizer default.
            **kwargs: Additional metadata key-value pairs stored in `self.metadata`.
        """
        super(OmniDatasetForTokenClassification, self).__init__(
            dataset_name_or_path, tokenizer, max_length, **kwargs
        )
        self.metadata.update(
            {
                "library_name": name,
                "omnigenbench_version": version,
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

    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for sequence classification.

        Args:
            dataset_name_or_path: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForSequenceClassification, self).__init__(
            dataset_name_or_path, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": name,
                "omnigenbench_version": version,
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
            labels = self.label2id.get(str(labels), -100)
            if not isinstance(labels, int):
                raise Exception(
                    "The label must be an integer for sequence classification."
                )
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

    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for token regression.

        Args:
            dataset_name_or_path: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForTokenRegression, self).__init__(
            dataset_name_or_path, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": name,
                "omnigenbench_version": version,
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

    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        """
        Initialize the dataset for sequence regression.

        Args:
            dataset_name_or_path: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.
        """
        super(OmniDatasetForSequenceRegression, self).__init__(
            dataset_name_or_path, tokenizer, max_length, **kwargs
        )

        self.metadata.update(
            {
                "library_name": name,
                "omnigenbench_version": version,
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


class OmniDatasetForMultiLabelClassification(OmniDataset):
    """
    Dataset class for multi-label classification tasks in genomics.

    This class extends `OmniDataset` to prepare input sequences and their corresponding
    multi-label targets. It's designed for tasks where each sequence can belong to
    multiple categories simultaneously, such as transcription factor binding prediction
    where a DNA sequence can bind to multiple transcription factors.

    Attributes:
        metadata: Dictionary containing dataset metadata including library information
        label_indices: Optional list of label indices to select specific labels from the dataset
    """

    def __init__(
        self,
        dataset_name_or_path,
        tokenizer,
        max_length=None,
        label_indices=None,
        **kwargs,
    ):
        """
        Initialize the dataset for multi-label classification.

        Args:
            dataset_name_or_path: Path to the data file or a list of paths.
                        Supported formats depend on the `OmniDataset` implementation.
            tokenizer: The tokenizer instance to use for converting sequences into
                      tokenized inputs.
            max_length: The maximum sequence length for tokenization. Sequences longer
                       than this will be truncated. If None, a default or tokenizer's
                       max length will be used.
            label_indices: Optional list of integers specifying which label indices to use.
                          If provided, only these labels will be selected from the full label vector.
                          Useful for focusing on specific tasks or reducing dimensionality.
            **kwargs: Additional keyword arguments to be stored in the dataset's metadata.

        Example:
            >>> # Load dataset with all labels
            >>> dataset = OmniDatasetForMultiLabelClassification(
            ...     "deepsea_data.json", tokenizer, max_length=1000
            ... )

            >>> # Load dataset with only specific transcription factors (indices 0, 5, 10)
            >>> dataset = OmniDatasetForMultiLabelClassification(
            ...     "deepsea_data.json", tokenizer, max_length=1000,
            ...     label_indices=[0, 5, 10]
            ... )
        """
        super(OmniDatasetForMultiLabelClassification, self).__init__(
            dataset_name_or_path, tokenizer, max_length, **kwargs
        )

        self.label_indices = label_indices

        self.metadata.update(
            {
                "library_name": name,
                "omnigenbench_version": version,
                "task": "genome_multi_label_classification",
            }
        )

        for key, value in kwargs.items():
            self.metadata[key] = value

    def prepare_input(self, instance, **kwargs):
        """
        Prepare a single data instance for multi-label classification.

        This method handles both string sequences and dictionary instances
        containing sequence and multi-label information. It tokenizes the input
        sequence and prepares multi-hot encoded labels for classification.

        Args:
            instance: A single data instance. Can be a string representing the sequence
                     or a dictionary with 'seq'/'sequence' and 'labels'/'label' keys.
                     For multi-label tasks, labels should be a list or array of binary values.
            **kwargs: Additional keyword arguments for tokenization, such as 'padding'
                     and 'truncation'.

        Returns:
            dict: A dictionary of tokenized inputs, including 'input_ids', 'attention_mask',
                  and 'labels' (tensor of multi-hot encoded labels).

        Raises:
            Exception: If the input instance format is unknown or if a dictionary
                      instance does not contain a 'seq' or 'sequence' key.

        Example:
            For an instance like:
            {
                "sequence": "ACGTAGCTAGCTAGCTAGC...",
                "label": [0, 1, 0, 0, 1, ..., 0],  # Multi-hot encoded labels for TF binding
            }
            Returns tokenized inputs with labels as float tensor for multi-label classification.
        """

        def truncate_sequence(seq, max_len):
            """
            Truncate sequence to max_len, centering the truncation if sequence is too long.
            Pad with 'N' if sequence is too short.
            """
            if max_len is None:
                return seq
            if len(seq) == max_len:
                return seq
            elif len(seq) > max_len:
                start_idx = (len(seq) - max_len) // 2
                return seq[start_idx : start_idx + max_len]
            else:
                return seq + ("N" * (max_len - len(seq)))

        labels = None
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

        # Truncate or pad sequence to desired length
        sequence = truncate_sequence(sequence, self.max_length)

        tokenized_inputs = self.tokenizer(
            sequence,
            padding=kwargs.get("padding", "do_not_pad"),
            truncation=kwargs.get("truncation", True),
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Squeeze tensors to remove batch dimension
        for key in tokenized_inputs:
            if (
                isinstance(tokenized_inputs[key], torch.Tensor)
                and tokenized_inputs[key].ndim > 1
            ):
                tokenized_inputs[key] = tokenized_inputs[key].squeeze(0)

        # Process multi-label targets
        labels_tensor = None
        if labels is not None:
            # Convert to tensor with float32 dtype for multi-label classification
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            # Select specific label indices if provided
            if hasattr(self, "label_indices") and self.label_indices is not None:
                labels_tensor = labels_tensor[
                    torch.tensor(self.label_indices, dtype=torch.long)
                ]

        tokenized_inputs["labels"] = labels_tensor
        return tokenized_inputs

    def print_label_statistics(self):
        """
        Print statistics about the multi-label distribution in the dataset.
        This includes the number of positive labels per sample and per label class.
        """
        if (
            not self.data
            or "labels" not in self.data[0]
            or self.data[0]["labels"] is None
        ):
            fprint("No labels found in the dataset.")
            return

        # Collect all labels
        all_labels = []
        for data_item in self.data:
            if data_item["labels"] is not None:
                labels = data_item["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                all_labels.append(labels)

        if not all_labels:
            fprint("No valid labels found in the dataset.")
            return

        all_labels = np.array(all_labels)
        num_samples, num_labels = all_labels.shape

        # Calculate statistics
        positive_per_sample = np.sum(all_labels, axis=1)
        positive_per_label = np.sum(all_labels, axis=0)

        fprint("\nMulti-Label Dataset Statistics:")
        fprint("-" * 50)
        fprint(f"Total samples: {num_samples}")
        fprint(f"Total label classes: {num_labels}")
        fprint(
            f"Average positive labels per sample: {np.mean(positive_per_sample):.2f}"
        )
        fprint(f"Min positive labels per sample: {np.min(positive_per_sample)}")
        fprint(f"Max positive labels per sample: {np.max(positive_per_sample)}")

        fprint("\nLabel Distribution:")
        fprint("-" * 50)
        fprint(f"{'Label Index':<12}\t{'Positive Count':<15}\t{'Percentage':<10}")
        fprint("-" * 50)

        for i, count in enumerate(positive_per_label):
            percentage = (count / num_samples) * 100
            fprint(f"{i:<12}\t\t{int(count):<15}\t\t{percentage:.2f}%")

        fprint("-" * 50)
