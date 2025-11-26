# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import os
import random
import warnings
import zipfile
import findfile
import requests
from collections import Counter

import numpy as np
import torch
import tqdm

from transformers import BatchEncoding

from ..misc.utils import fprint, env_meta_info, RNA2StructureCache


def covert_input_to_tensor(data):
    """
    This function traverses through nested dictionaries and lists, converting
    numerical values to PyTorch tensors while preserving the structure.

    Args:
        data (list or dict): A list or dictionary containing data samples.

    Returns:
        list or dict: The data structure with numerical values converted to tensors.

    Example:
        >>> data = [{'input_ids': [1, 2, 3], 'labels': [0]}]
        >>> tensor_data = covert_input_to_tensor(data)
        >>> print(type(tensor_data[0]['input_ids']))  # <class 'torch.Tensor'>
    """
    for d in data:
        if isinstance(d, dict) or isinstance(d, BatchEncoding):
            for key, value in d.items():
                try:
                    if not isinstance(value, torch.Tensor):
                        d[key] = torch.tensor(value)
                except Exception as e:
                    pass
        elif isinstance(d, list):
            for value in d:
                covert_input_to_tensor(value)
            covert_input_to_tensor(d)

    return data


class OmniDict(dict):
    """
    This class extends the standard Python dictionary to provide a convenient
    method for moving all tensor values to a specific device (CPU/GPU).
    """

    def __init__(self, *args, **kwargs):
        super(OmniDict, self).__init__(*args, **kwargs)

    def to(self, device):
        """
        Moves all tensor values in the dictionary to the specified device.

        Args:
            device (str or torch.device): The target device (e.g., 'cuda:0' or 'cpu').

        Returns:
            OmniDict: The dictionary itself, with tensors moved to the new device.

        Example:
            >>> data = OmniDict({'input_ids': torch.tensor([1, 2, 3])})
            >>> data.to('cuda:0')  # Moves tensors to GPU
        """
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class OmniDataset(torch.utils.data.Dataset):
    """
    Abstract base class providing a unified interface for genomic datasets in the OmniGenBench
    framework. This class handles polymorphic data loading from multiple formats, integrated
    tokenization, label management, and PyTorch DataLoader compatibility.

    **Design Pattern**: This class implements the Strategy pattern for format-specific parsing
    while maintaining a consistent API. Different file formats (JSON, CSV, FASTA, Parquet, etc.)
    are handled transparently through pluggable loaders, with tokenization and preprocessing
    applied uniformly regardless of input format.

    **Key Features**:

    - **Format Agnosticism**: Supports JSON, CSV, Parquet, FASTA, FASTQ, BED, VCF, and NumPy
      formats through auto-detection based on file extension. Custom formats can be added by
      subclassing and implementing format-specific loaders.

    - **Integrated Tokenization**: Sequences are tokenized within the dataset pipeline for
      consistency and efficient caching. Tokenization parameters (max_length, padding, truncation)
      are configured at dataset initialization.

    - **Lazy Loading**: Large datasets are loaded incrementally to minimize memory footprint.
      Data is read into memory on-demand during training/inference rather than all at once.

    - **Label Management**: Automatic bidirectional mapping between string labels and integer
      indices (label2id/id2label), with support for multi-label scenarios and PyTorch's -100
      ignore convention for masked tokens.

    - **RNA Structure Integration**: Optional secondary structure prediction via ViennaRNA
      for structure-aware models. Structures are cached to avoid redundant computation.

    - **Sequence Filtering**: Optional filtering of sequences exceeding max_length via
      drop_long_seq parameter, useful for maintaining fixed-length batches without truncation.

    **Data Format Convention**: All input files must contain at minimum a ``sequence`` field
    (or one of its aliases: ``seq``, ``text``, ``dna``, ``rna``). For supervised tasks, a
    ``label`` field (or aliases: ``labels``, ``target``, ``y``) is also required. Additional
    custom fields are preserved and passed through the pipeline.

    **Supported File Formats**:

    - **JSON**: Line-delimited JSON (.json, .jsonl) with one record per line
    - **CSV/TSV**: Comma or tab-separated values (.csv, .tsv) with header row
    - **Parquet**: Apache Parquet format (.parquet) for efficient columnar storage
    - **FASTA**: Biological sequence format (.fasta, .fa) with optional metadata in headers
    - **FASTQ**: Sequencing format (.fastq, .fq) with quality scores (quality scores ignored)
    - **BED**: Genomic interval format (.bed) for position-based features
    - **VCF**: Variant Call Format (.vcf) for genetic variants (experimental)
    - **NumPy**: NumPy arrays (.npy, .npz) for pre-computed features

    Attributes:
        tokenizer: Tokenizer instance for sequence encoding. Must be compatible with the
            model architecture being used. Can be OmniTokenizer or HuggingFace tokenizer.

        max_length (int): Maximum sequence length for tokenization. Sequences exceeding this
            length are truncated (default) or dropped (if drop_long_seq=True).

        label2id (dict): Mapping from string labels to integer indices. Automatically populated
            during data loading if not provided. Example: {"negative": 0, "positive": 1}.

        id2label (dict): Inverse mapping from integer indices to string labels. Automatically
            generated from label2id.

        shuffle (bool): Whether to shuffle dataset order on initialization. Default True.
            Set to False for validation/test sets to maintain reproducible evaluation.

        structure_in (bool): Whether to include RNA secondary structure predictions as input
            features. Requires ViennaRNA installation. Default False. Adds dot-bracket notation
            as additional input for structure-aware models.

        drop_long_seq (bool): Whether to drop sequences longer than max_length instead of
            truncating them. Default False. When True, sequences exceeding max_length are
            filtered out during loading.

        metadata (dict): Framework metadata including version information and environment details.
            Automatically populated with Python version, OmniGenBench version, timestamp, etc.

        rna2structure (RNA2StructureCache): Persistent cache for RNA structure predictions to
            avoid redundant ViennaRNA calls. Only created when structure_in=True.

        data (list): Internal storage for loaded dataset samples. Each element is a dictionary
            containing 'sequence', 'label', and any additional custom fields.

    Note:
        This is an abstract base class. Use task-specific subclasses for actual datasets:

        - ``OmniDatasetForSequenceClassification``: Sequence-level classification
        - ``OmniDatasetForMultiLabelClassification``: Multi-label classification
        - ``OmniDatasetForTokenClassification``: Per-nucleotide classification
        - ``OmniDatasetForSequenceRegression``: Sequence-level regression
        - ``OmniDatasetForTokenRegression``: Per-nucleotide regression
    """

    def __init__(
        self, dataset_name_or_path=None, tokenizer=None, max_length=None, **kwargs
    ):
        """
        Initializes the genomic dataset with flexible input sources and preprocessing options.

        This method handles dataset loading from various sources: local file paths, HuggingFace
        Hub identifiers, or lists of file paths for multi-file datasets. It automatically
        detects file formats and applies appropriate parsers.

        Args:
            dataset_name_or_path (str or list): One of the following:
                - Path to a single data file (e.g., "data.json", "sequences.fasta")
                - HuggingFace Hub dataset identifier (e.g., "yangheng/tfb_promoters")
                - List of file paths for multi-file datasets
                - Directory path containing train.json/test.json/val.json
            tokenizer: Tokenizer instance for sequence encoding. Must implement the
                OmniTokenizer interface or be a HuggingFace tokenizer.
            max_length (int, optional): Maximum sequence length after tokenization.
                Sequences exceeding this length are truncated unless drop_long_seq=True.
                If None, uses tokenizer's default max_length or 512.
            **kwargs: Additional keyword arguments:
                - label2id (dict): Pre-defined mapping from labels to integer IDs. If not
                  provided, will be auto-generated from unique labels in the dataset.
                - shuffle (bool): Whether to shuffle dataset order. Defaults to True.
                  Set to False for validation/test sets to maintain consistent ordering.
                - structure_in (bool): Whether to include RNA secondary structure
                  predictions as additional input features. Requires ViennaRNA installation.
                  Defaults to False. Adds computational overhead during loading.
                - drop_long_seq (bool): Whether to filter out sequences longer than
                  max_length instead of truncating. Defaults to False. Useful for
                  maintaining fixed-length training batches without truncation artifacts.
                - dataset_url (str): URL to download dataset if not found locally.
                  Supports .zip archives that will be automatically extracted.
                - cache_dir (str): Directory for caching downloaded datasets. Defaults
                  to "./__OMNIGENBENCH_DATA__/datasets/".
                - dataset_name_or_path (str): Alternative parameter name for dataset_name_or_path
                  for backward compatibility.

        Raises:
            ValueError: If dataset_name_or_path is not provided or cannot be located.
            FileNotFoundError: If the specified dataset file does not exist and cannot
                be downloaded from the provided dataset_url.
            RuntimeError: If the dataset format is not recognized or parsing fails.

        Example:
            >>> # Pattern 1: Initialize with a single local file (format auto-detected)
            >>> dataset = OmniDatasetForSequenceClassification(
            ...     "promoters.json",
            ...     tokenizer=tokenizer,
            ...     max_length=512
            ... )

            >>> # Pattern 2: Initialize with explicit label mapping
            >>> dataset = OmniDatasetForSequenceClassification(
            ...     "data.csv",
            ...     tokenizer=tokenizer,
            ...     label2id={"positive": 1, "negative": 0},
            ...     max_length=256
            ... )

            >>> # Pattern 3: Initialize with automatic dataset download from URL
            >>> dataset = OmniDatasetForSequenceClassification(
            ...     "custom_dataset.zip",
            ...     tokenizer=tokenizer,
            ...     dataset_url="https://example.com/datasets/custom_dataset.zip",
            ...     cache_dir="./my_datasets/"
            ... )

            >>> # Pattern 4: Load with RNA secondary structure features
            >>> dataset = OmniDatasetForSequenceClassification(
            ...     "rna_sequences.json",
            ...     tokenizer=tokenizer,
            ...     structure_in=True,  # Adds ViennaRNA structure predictions
            ...     max_length=512
            ... )

            >>> # Pattern 5: Load from multiple files
            >>> dataset = OmniDatasetForSequenceClassification(
            ...     ["train_part1.json", "train_part2.json", "train_part3.json"],
            ...     tokenizer=tokenizer,
            ...     shuffle=True
            ... )
        """
        super(OmniDataset, self).__init__()
        if not dataset_name_or_path and kwargs.get("dataset_name_or_path", None):
            dataset_name_or_path = kwargs.get("dataset_name_or_path", None)
        if not dataset_name_or_path:
            raise ValueError("Please provide dataset_name_or_path")

        self.metadata = env_meta_info()
        self.dataset_info = None  # Store dataset_info separately from metadata
        self.tokenizer = tokenizer
        self.label2id = kwargs.get("label2id", None)
        self.shuffle = kwargs.get("shuffle", True)
        self.structure_in = kwargs.get("structure_in", False)
        self.drop_long_seq = kwargs.get("drop_long_seq", False)
        self.force_padding = kwargs.get("force_padding", True)
        if self.structure_in and not hasattr(self, "rna2structure"):
            self.rna2structure = RNA2StructureCache()

        if self.label2id is not None:
            self.id2label = {v: k for k, v in self.label2id.items()}
        else:
            fprint(
                "No label2id provided, something wrong may happen. "
                "label2id indicates the mapping from label to indices. "
                "e.g., {'positive': 1, 'negative': 0} for binary classification."
            )

        if max_length is not None:
            fprint(
                f"Detected max_length={max_length} in the dataset, using it as the max_length."
            )
            self.max_length = max_length
        elif (
            hasattr(self.tokenizer, "max_length")
            and self.tokenizer.max_length is not None
        ):
            fprint(
                f"Detected max_length={self.tokenizer.max_length} from the tokenizer."
            )
            self.max_length = self.tokenizer.max_length
        else:
            fprint(f"No max_length detected, using default max_length=512.")
            self.max_length = 512

        self.tokenizer.max_length = self.max_length
        self.examples = []
        self.data = []

        if dataset_name_or_path is not None:
            # Check if dataset needs to be downloaded
            fprint(f"Loading data from {dataset_name_or_path}...")
            self.load_dataset_name_or_path(dataset_name_or_path, **kwargs)
            # Try to load dataset_info.json from the same directory
            self._load_dataset_info(dataset_name_or_path)
            self._preprocessing()
            if self.tokenizer is not None:
                for example in tqdm.tqdm(self.examples):
                    if hasattr(self.tokenizer, "max_length"):
                        self.tokenizer.max_length = self.max_length
                    else:
                        self.tokenizer.base_tokenizer.max_length = self.max_length

                    import inspect

                    new_args = {}
                    tokenization_args = inspect.getfullargspec(
                        self.tokenizer.encode
                    ).args
                    for key in kwargs:
                        if key in tokenization_args:
                            new_args[key] = kwargs[key]
                    prepared_input = self.prepare_input(example, **new_args)

                    if not prepared_input:
                        continue

                    if (
                        self.drop_long_seq
                        and len(prepared_input["input_ids"]) > self.max_length
                    ):
                        fprint(
                            f"Dropping sequence {example['sequence']} due to length > {self.max_length}"
                        )
                    else:

                        if isinstance(prepared_input, BatchEncoding):
                            prepared_input = [prepared_input]

                        # Squeeze the batch dimension if it exists
                        for item in prepared_input:
                            for key, value in item.items():
                                item[key] = value.squeeze(0)
                            self.data.append(item)

                self._postprocessing()
                self._pad_and_truncate()

    def get_dataloader(
        self, batch_size=16, shuffle=None, num_workers=0, pin_memory=None, **kwargs
    ):
        """
        Creates a PyTorch DataLoader for this dataset.

        Args:
            batch_size (int): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data. If None, uses self.shuffle.
            num_workers (int): Number of worker processes for data loading.
            pin_memory (bool): Whether to pin memory. If None, auto-detects based on CUDA availability.
            **kwargs: Additional arguments passed to DataLoader.

        Returns:
            torch.utils.data.DataLoader: A DataLoader for this dataset.
        """
        if shuffle is None:
            shuffle = self.shuffle

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        from torch.utils.data import DataLoader

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs,
        )

    @classmethod
    def from_hub(
        cls,
        dataset_name_or_path,
        tokenizer,
        splits=None,
        max_length=None,
        cache_dir=None,
        **kwargs,
    ):
        """
        Create OmniDataset instances from HuggingFace Hub or local directory.

        This method supports loading datasets from:
        1. OmniGenBench Hub on HuggingFace (downloads if needed)
        2. Local directory containing dataset files

        Args:
            dataset_name_or_path (str): Name of the dataset on HuggingFace Hub, or path to local directory.
            tokenizer: The tokenizer to use for processing sequences.
            splits (list, optional): List of splits to create. Defaults to ['train', 'valid', 'test'].
            max_length (int, optional): Maximum sequence length.
            cache_dir (str, optional): Directory to cache the dataset or look for local files.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            dict: Dictionary containing datasets for each split.

        Example:
            >>> from omnigenbench import OmniTokenizer, OmniDatasetForSequenceClassification
            >>> tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-52M")

            >>> # Load from HuggingFace Hub
            >>> datasets = OmniDatasetForSequenceClassification.from_hub(
            ...     "translation_efficiency_prediction",
            ...     tokenizer=tokenizer
            ... )

            >>> # Load from local directory
            >>> datasets = OmniDatasetForSequenceClassification.from_hub(
            ...     "/path/to/local/dataset",
            ...     tokenizer=tokenizer,
            ...     cache_dir="/path/to/local/dataset"
            ... )

            >>> train_loader = datasets['train'].get_dataloader(batch_size=16)
        """
        if splits is None:
            splits = ["train", "valid", "test"]

        # Determine if dataset_name_or_path is a local path or a HuggingFace dataset name
        is_local = os.path.exists(dataset_name_or_path) or (
            cache_dir and os.path.exists(cache_dir)
        )

        if not is_local:
            # Download from HuggingFace if not a local path
            from ...src.utility.hub_utils import download_dataset

            cache_dir = download_dataset(
                dataset_name_or_path,
                cache_dir=cache_dir,
                use_hf_api=True,
                force_download=kwargs.get("force_download", False),
            )
        else:
            fprint(
                f"Loading dataset from local path: {dataset_name_or_path or cache_dir}"
            )

        # Create datasets for each split
        datasets = {"train": None, "valid": None, "test": None}
        keys_to_search = {
            "train": ["train."],
            "valid": ["val.", "valid.", "dev."],
            "test": ["test."],
        }
        for split in splits:
            if not cache_dir:
                if is_local and os.path.exists(dataset_name_or_path):
                    # If dataset_name_or_path itself is a local directory
                    cache_dir = dataset_name_or_path
                else:
                    # Use default cache directory
                    cache_dir = os.getcwd()

            dataset_name_or_path = findfile.find_files(
                cache_dir,
                or_key=keys_to_search[split],
                exclude_key=[".ipynb", ".py", "md", "txt"],
            )
            if not dataset_name_or_path:
                fprint(
                    f"Warning: No data files found for split '{split}' in {cache_dir}. Skipping this split."
                )
                continue
            else:
                fprint(f"Load data files for split '{split}': {dataset_name_or_path}")

            datasets[split] = cls(
                dataset_name_or_path=dataset_name_or_path,
                tokenizer=tokenizer,
                max_length=max_length,
                split=split,
                **kwargs,
            )

        return datasets

    @classmethod
    def from_huggingface(
        cls,
        dataset_name_or_path,
        tokenizer,
        splits=None,
        max_length=None,
        cache_dir=None,
        **kwargs,
    ):
        """
        Create OmniDataset instances from a HuggingFace dataset.

        .. deprecated:: 0.3.0
            `from_huggingface` is deprecated and will be removed in version 0.4.0.
            Use `from_hub` instead, which supports both HuggingFace Hub and local data sources.

        Args:
            dataset_name_or_path (str): Name of the HuggingFace dataset or base URL.
            tokenizer: The tokenizer to use for processing sequences.
            splits (list, optional): List of splits to create. Defaults to ['train', 'valid', 'test'].
            max_length (int, optional): Maximum sequence length.
            cache_dir (str, optional): Directory to cache the dataset.
            **kwargs: Additional arguments passed to the dataset constructor.

        Returns:
            dict: Dictionary containing datasets for each split.

        Example:
            >>> from omnigenbench import OmniTokenizer,OmniDatasetForSequenceClassification
            >>> tokenizer = OmniTokenizer.from_pretrained("yangheng/OmniGenome-52M")
            >>> datasets = OmniDatasetForSequenceClassification.from_huggingface(
            ...     "translation_efficiency_prediction",
            ...     tokenizer=tokenizer
            ... )
            >>> train_loader = datasets['train'].get_dataloader(batch_size=16)
        """
        warnings.warn(
            "from_huggingface() is deprecated and will be removed in version 0.4.0. "
            "Please use from_hub() instead, which supports both HuggingFace Hub and local data sources.",
            DeprecationWarning,
            stacklevel=2,
        )

        return cls.from_hub(
            dataset_name_or_path=dataset_name_or_path,
            tokenizer=tokenizer,
            splits=splits,
            max_length=max_length,
            cache_dir=cache_dir,
            **kwargs,
        )

    def print_label_distribution(self):
        """
        Print the distribution of labels for 0-dimensional (scalar) labels.
        This is useful for classification tasks where each sample has a single label.
        """
        # Check if we have scalar labels
        if self.data and "labels" in self.data[0]:
            first_label = self.data[0]["labels"]
            if isinstance(first_label.item(), float):
                return

            if not isinstance(first_label, torch.Tensor) or first_label.ndim == 0:
                # Convert labels to list of integers
                labels = [int(d["labels"]) for d in self.data]

                # Count frequency of each label
                label_counts = Counter(labels)
                total_samples = len(labels)

                # Sort by label value
                sorted_counts = sorted(label_counts.items())

                fprint("\nLabel Distribution:")
                fprint("-" * 40)
                fprint(f"{'Label':<10}\t\t{'Count':<10}\t\t{'Percentage':<10}")
                fprint("-" * 40)

                for label, count in sorted_counts:
                    percentage = (count / total_samples) * 100
                    label_name = (
                        self.id2label[label]
                        if hasattr(self, "id2label")
                        else str(label)
                    )
                    fprint(f"{label_name:<10}\t\t{count:<10}\t\t{percentage:.2f}%")

                fprint("-" * 40)
                fprint(f"Total samples: {total_samples}")
            else:
                fprint(
                    "Warning: This method is only for scalar (0-dimensional) labels."
                )
        else:
            fprint("No labels found in the dataset.")

    def to(self, device):
        """
        Moves all tensor data in the dataset to the specified device.

        Args:
            device (str or torch.device): The target device.

        Returns:
            OmniDataset: The dataset itself.
        """
        for data_item in self.data:
            for key, value in data_item.items():
                if isinstance(value, torch.Tensor):
                    data_item[key] = value.to(device)
        return self

    def _pad_and_truncate(self, pad_value=0):
        """
        Pads and truncates sequences in the dataset to a uniform length.
        The length is determined dynamically based on the longest sequence in the batch,
        up to the `self.max_length` limit, and adjusted to be a multiple of 8.

        Args:
            pad_value (int, optional): The value to use for padding. Defaults to 0.

        Returns:
            list: The padded and truncated data.
        """
        key_lengths = {key: [] for key in self.data[0].keys()}
        for item in self.data:
            for key, value in item.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                length = value.size(0) if value.ndim > 0 else 0
                key_lengths[key].append(length)

        skip_padding_for_key = {
            key: len(set(lengths)) == 1 for key, lengths in key_lengths.items()
        }

        skipped_keys = [key for key, skip in skip_padding_for_key.items() if skip]
        if len(skipped_keys) == len(self.data[0].keys()) and not self.force_padding:
            fprint(
                "All keys have consistent sequence lengths, skipping padding and truncation."
            )
            return self.data

        if hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.base_tokenizer.pad_token_id

        max_input_length = max(
            [
                torch.sum(data_item["input_ids"] != pad_token_id).item()
                for data_item in self.data
            ]
        )
        max_label_length = max(
            [
                (data_item["labels"].shape[0] if data_item["labels"].ndim >= 1 else 0)
                for data_item in self.data
            ]
        )

        original_max_length = max(max_input_length, max_label_length)
        original_max_length = min(original_max_length, self.max_length)

        remainder = original_max_length % 8
        if remainder != 0:
            adjusted_max_length = original_max_length + (8 - remainder)
            adjusted_max_length = min(adjusted_max_length, self.max_length)
        else:
            adjusted_max_length = original_max_length
        max_length = adjusted_max_length

        first_labels = self.data[0]["labels"]

        label_shape = first_labels.shape
        if len(label_shape) >= 1:
            max_length = max(max_length, self.data[0]["labels"].shape[0])
            label_padding_length = max(max_label_length, max_length)
            max_length = max(max_length, label_padding_length)
        else:
            label_padding_length = 0

        fprint(
            f"Max sequence length updated -> Reset max_length={max_length},"
            f" label_padding_length={label_padding_length}"
        )

        for data_item in self.data:
            for key, value in data_item.items():
                dtype = value.dtype
                if key in skipped_keys:
                    data_item[key] = data_item[key].to(dtype)
                    continue
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                if "label" in key and (
                    value.dtype == torch.int16 or value.dtype == torch.int32
                ):
                    data_item[key] = value.long()

                if "label" in key:
                    if value.ndim == 0:
                        padding_length = 0
                    else:
                        padding_length = label_padding_length - value.size(0)
                else:
                    padding_length = max_length - value.size(0)

                if padding_length > 0:

                    if key == "input_ids":
                        _pad_value = pad_token_id
                    elif key == "attention_mask":
                        _pad_value = 0
                    elif "ids" in key:
                        _pad_value = 0
                    elif "label" in key:
                        _pad_value = -100
                    elif "ids" in key:
                        _pad_value = pad_token_id
                    else:
                        _pad_value = pad_value

                    if value.ndim == 2:
                        pad_shape = (padding_length, value.size(1))
                    else:
                        pad_shape = (padding_length,)
                    pad_tensor = torch.full(pad_shape, _pad_value, dtype=dtype)
                    data_item[key] = torch.cat([value, pad_tensor], dim=0)
                elif padding_length < 0:
                    data_item[key] = value[:max_length]

                data_item[key] = data_item[key].to(dtype)

        return self.data

    def _load_dataset_info(self, dataset_path):
        """
        Load dataset_info.json from the dataset directory.

        This method searches for dataset_info.json in the directory containing
        the dataset file(s). The dataset_info provides structured metadata about
        the dataset including description, statistics, features, and more.

        Note: This is separate from the 'metadata' attribute used by OmniGenBench
        for model/tokenizer versioning. dataset_info is specifically for dataset
        documentation and characteristics.

        Args:
            dataset_path (str or list): Path to dataset file(s) or directory
        """
        import json
        import os

        # Determine the directory to search
        if isinstance(dataset_path, list):
            # Use the directory of the first file
            search_dir = os.path.dirname(os.path.abspath(dataset_path[0]))
        elif os.path.isdir(dataset_path):
            search_dir = dataset_path
        else:
            search_dir = os.path.dirname(os.path.abspath(dataset_path))

        # Look for dataset_info.json in the directory
        info_path = os.path.join(search_dir, "dataset_info.json")

        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    self.dataset_info = json.load(f)
                fprint(f"✓ Loaded dataset_info from: {info_path}")
            except Exception as e:
                fprint(f"Warning: Failed to load dataset_info.json: {e}")
                self.dataset_info = None
        else:
            fprint(f"Note: No dataset_info.json found in {search_dir}")
            self.dataset_info = None

    def info(self, sections=None, detailed=False, return_dict=False):
        """
        Print formatted dataset information in table format using tabulate.

        This method displays dataset_info in a human-readable table format. It's separate
        from model/tokenizer metadata and focuses on dataset characteristics.

        Args:
            sections (list, optional): List of section names to print. If None, prints all.
                Available sections: 'basic', 'statistics', 'features', 'splits',
                'preprocessing', 'metrics', 'citation', 'all'
            detailed (bool, optional): If True, prints detailed JSON data. Defaults to False.
            return_dict (bool, optional): If True, returns the dataset_info dict instead of printing.
                Defaults to False.

        Returns:
            dict or None: Returns dataset_info dict if return_dict=True, otherwise None.

        Example:
            >>> dataset.info()  # Print all sections in table format
            >>> dataset.info(sections=['basic', 'statistics'])  # Print specific sections
            >>> dataset.info(detailed=True)  # Print detailed JSON data
            >>> info_dict = dataset.info(return_dict=True)  # Get info as dictionary
        """
        import json

        try:
            from tabulate import tabulate
        except ImportError:
            fprint("Warning: 'tabulate' package not found. Installing...")
            import subprocess
            import sys

            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
            from tabulate import tabulate

        if self.dataset_info is None:
            fprint(
                "No dataset_info available. Load a dataset with dataset_info.json or use load_dataset_info()."
            )
            return None

        # Return dict if requested
        if return_dict:
            return self.dataset_info

        # Print detailed JSON if requested
        if detailed:
            fprint("\n" + "=" * 80)
            fprint("DETAILED DATASET INFORMATION (JSON)".center(80))
            fprint("=" * 80 + "\n")
            fprint(json.dumps(self.dataset_info, indent=2, ensure_ascii=False))
            fprint("\n" + "=" * 80 + "\n")
            return self.dataset_info

        if sections is None or (isinstance(sections, list) and "all" in sections):
            sections = [
                "basic",
                "statistics",
                "features",
                "splits",
                "preprocessing",
                "metrics",
                "citation",
            ]

        info = self.dataset_info

        fprint("\n" + "=" * 80)
        fprint("DATASET INFORMATION".center(80))
        fprint("=" * 80)

        # Basic Information - Table Format
        if "basic" in sections:
            fprint("\n📊 BASIC INFORMATION")
            basic_fields = [
                ("dataset_name", "Dataset Name"),
                ("dataset_version", "Version"),
                ("description", "Description"),
                ("task_type", "Task Type"),
                ("domain", "Domain"),
                ("source", "Source"),
                ("license", "License"),
            ]

            table_data = []
            for field_key, field_label in basic_fields:
                if field_key in info:
                    table_data.append([field_label, info[field_key]])

            if table_data:
                fprint(
                    tabulate(
                        table_data,
                        headers=["Field", "Value"],
                        tablefmt="grid",
                        maxcolwidths=[20, 60],
                    )
                )

        # Statistics - Table Format
        if "statistics" in sections and "statistics" in info:
            fprint("\n📈 STATISTICS")
            stats = info["statistics"]

            # Separate simple and nested statistics
            simple_stats = {k: v for k, v in stats.items() if not isinstance(v, dict)}
            nested_stats = {k: v for k, v in stats.items() if isinstance(v, dict)}

            # Print simple statistics in table
            if simple_stats:
                table_data = [
                    [k.replace("_", " ").title(), v] for k, v in simple_stats.items()
                ]
                fprint(
                    tabulate(
                        table_data,
                        headers=["Metric", "Value"],
                        tablefmt="grid",
                        maxcolwidths=[30, 50],
                    )
                )

            # Print nested statistics
            if nested_stats:
                for key, value in nested_stats.items():
                    formatted_key = key.replace("_", " ").title()
                    fprint(f"\n  {formatted_key}:")
                    table_data = [
                        [sub_key, sub_value] for sub_key, sub_value in value.items()
                    ]
                    fprint(
                        tabulate(
                            table_data,
                            headers=["Item", "Value"],
                            tablefmt="grid",
                            maxcolwidths=[25, 50],
                        )
                    )

        # Features - Table Format
        if "features" in sections and "features" in info:
            fprint("\n🔧 FEATURES")
            features = info["features"]

            if "input" in features:
                fprint("\n  Input Features:")
                table_data = []
                for feat_name, feat_info in features["input"].items():
                    if isinstance(feat_info, dict):
                        feat_type = feat_info.get("type", "N/A")
                        feat_desc = feat_info.get("description", "N/A")
                        table_data.append([feat_name, feat_type, feat_desc])
                    else:
                        table_data.append([feat_name, "N/A", "N/A"])

                if table_data:
                    fprint(
                        tabulate(
                            table_data,
                            headers=["Feature", "Type", "Description"],
                            tablefmt="grid",
                            maxcolwidths=[25, 15, 40],
                        )
                    )

            if "output" in features:
                fprint("\n  Output Features:")
                table_data = []
                for feat_name, feat_info in features["output"].items():
                    if isinstance(feat_info, dict):
                        feat_type = feat_info.get("type", "N/A")
                        feat_desc = feat_info.get("description", "N/A")
                        table_data.append([feat_name, feat_type, feat_desc])
                    else:
                        table_data.append([feat_name, "N/A", "N/A"])

                if table_data:
                    fprint(
                        tabulate(
                            table_data,
                            headers=["Feature", "Type", "Description"],
                            tablefmt="grid",
                            maxcolwidths=[25, 15, 40],
                        )
                    )

        # Data Splits - Table Format
        if "splits" in sections and "data_splits" in info:
            fprint("\n📂 DATA SPLITS")

            # Collect all unique keys across splits
            all_keys = set()
            for split_info in info["data_splits"].values():
                if isinstance(split_info, dict):
                    all_keys.update(split_info.keys())

            # Build table
            if all_keys:
                headers = ["Split"] + [
                    key.replace("_", " ").title() for key in sorted(all_keys)
                ]
                table_data = []

                for split_name, split_info in info["data_splits"].items():
                    if isinstance(split_info, dict):
                        row = [split_name.title()]
                        for key in sorted(all_keys):
                            row.append(split_info.get(key, "N/A"))
                        table_data.append(row)
                    else:
                        table_data.append([split_name.title(), split_info])

                if table_data:
                    fprint(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Preprocessing - Table Format
        if "preprocessing" in sections and "preprocessing" in info:
            fprint("\n⚙️  PREPROCESSING")
            preproc = info["preprocessing"]

            table_data = []
            for key, value in preproc.items():
                formatted_key = key.replace("_", " ").title()
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                else:
                    value_str = str(value)
                table_data.append([formatted_key, value_str])

            if table_data:
                fprint(
                    tabulate(
                        table_data,
                        headers=["Step", "Details"],
                        tablefmt="grid",
                        maxcolwidths=[30, 50],
                    )
                )

        # Evaluation Metrics - Table Format
        if "metrics" in sections and "evaluation_metrics" in info:
            fprint("\n📊 EVALUATION METRICS")
            metrics = info["evaluation_metrics"]

            table_data = []
            if "primary" in metrics:
                table_data.append(["Primary Metric", metrics["primary"]])
            if "description" in metrics:
                table_data.append(["Description", metrics["description"]])
            if "secondary" in metrics:
                secondary_str = (
                    ", ".join(metrics["secondary"])
                    if isinstance(metrics["secondary"], list)
                    else str(metrics["secondary"])
                )
                table_data.append(["Secondary Metrics", secondary_str])

            if table_data:
                fprint(
                    tabulate(
                        table_data,
                        headers=["Metric Type", "Details"],
                        tablefmt="grid",
                        maxcolwidths=[25, 55],
                    )
                )

        # Citation
        if "citation" in sections and "citation" in info:
            fprint("\n📚 CITATION")
            fprint("-" * 80)
            fprint(info["citation"])

        # Additional Notes
        if "notes" in info and (sections is None or "notes" in sections):
            fprint("\n📝 NOTES")
            fprint("-" * 80)
            for i, note in enumerate(info["notes"], 1):
                fprint(f"  {i}. {note}")

        fprint("\n" + "=" * 80 + "\n")

        return self.dataset_info

    def load_dataset_name_or_path(self, dataset_name_or_path, **kwargs):
        """
        Loads data from a file or list of files.

        Args:
            dataset_name_or_path (str or list): Path to the data file or a list of paths.
            **kwargs: Additional keyword arguments, e.g., `max_examples`.

        Returns:
            list: A list of examples.
        """
        examples = []
        max_examples = kwargs.get("max_examples", None)
        columns = kwargs.get("select_columns", None)
        if not isinstance(dataset_name_or_path, list):
            dataset_name_or_path = [dataset_name_or_path]

        for dataset_name_or_path in dataset_name_or_path:
            _examples = []

            if dataset_name_or_path.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(dataset_name_or_path, low_memory=False)
                for i in range(len(df)):
                    _examples.append(df.iloc[i].to_dict())
            elif dataset_name_or_path.endswith(
                ".json"
            ) or dataset_name_or_path.endswith(".jsonl"):
                import json

                try:
                    with open(dataset_name_or_path, "r", encoding="utf8") as f:
                        _examples = json.load(f)
                except:
                    with open(dataset_name_or_path, "r", encoding="utf8") as f:
                        lines = f.readlines()  # Assume the data is a list of examples
                    for i in range(len(lines)):
                        try:
                            lines[i] = json.loads(lines[i])
                        except:
                            print(lines[i])
                    for line in lines:
                        _examples.append(line)
            elif dataset_name_or_path.endswith(".parquet"):
                import pandas as pd

                df = pd.read_parquet(dataset_name_or_path)
                for i in range(len(df)):
                    _examples.append(df.iloc[i].to_dict())
            elif (dataset_name_or_path.endswith(".npy")
                  or dataset_name_or_path.endswith(".npz")):
                import numpy as np

                if dataset_name_or_path.endswith(".npy"):
                    data = np.load(dataset_name_or_path, allow_pickle=True)
                    if isinstance(data, np.ndarray):
                        for item in data:
                            _examples.append(
                                {
                                    "sequence": item["sequence"],
                                    "label": item.get("label", None),
                                }
                            )
                    else:
                        raise ValueError(
                            "Unexpected data format in .npy file, expected an array of dictionaries. e.g.,"
                            " [{'sequence': 'ATCG', 'label': 1}, ...]"
                        )
                elif dataset_name_or_path.endswith(".npz"):
                    data = np.load(dataset_name_or_path, allow_pickle=True)
                    for key in data.files:
                        item = data[key]
                        if isinstance(item, np.ndarray):
                            for sub_item in item:
                                _examples.append(
                                    {
                                        "sequence": sub_item["sequence"],
                                        "label": sub_item.get("label", None),
                                    }
                                )
                        else:
                            raise ValueError(
                                "Unexpected data format in .npz file, expected an array of dictionaries. e.g.,"
                                " [{'sequence': 'ATCG', 'label': 1}, ...]"
                            )
            elif dataset_name_or_path.endswith(
                (".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn")
            ):
                try:
                    from Bio import SeqIO
                except ImportError as e:
                    raise ImportError(
                        "Biopython is required for FASTA file parsing. "
                        "Please install it with: pip install biopython"
                    ) from e
                for record in SeqIO.parse(dataset_name_or_path, "fasta"):
                    _examples.append(
                        {
                            "id": record.id,
                            "sequence": str(record.seq),
                            "description": record.description,
                        }
                    )
            elif dataset_name_or_path.endswith((".fastq", ".fq")):
                try:
                    from Bio import SeqIO
                except ImportError as e:
                    raise ImportError(
                        "Biopython is required for FASTQ file parsing. "
                        "Please install it with: pip install biopython"
                    ) from e
                for record in SeqIO.parse(dataset_name_or_path, "fastq"):
                    _examples.append(
                        {
                            "id": record.id,
                            "sequence": str(record.seq),
                            "quality": record.letter_annotations.get(
                                "phred_quality", []
                            ),
                        }
                    )
            elif dataset_name_or_path.endswith(".bed"):
                import pandas as pd

                df = pd.read_csv(dataset_name_or_path, sep="\t", comment="#")
                for i in range(len(df)):
                    _examples.append(df.iloc[i].to_dict())

            else:
                raise Exception("Unknown file format.")

            if columns is not None:
                fprint(f"Selecting columns: {columns}")
                filtered_examples = []
                for ex in _examples:
                    filtered_ex = {col: ex[col] for col in columns if col in ex}
                    filtered_examples.append(filtered_ex)
                _examples = filtered_examples

            examples.extend(_examples)
            del _examples

            fprint(
                f"Reading from {dataset_name_or_path}, Loaded {len(examples)} examples so far..."
            )

        if self.shuffle is True:
            fprint("Detected shuffle=True, shuffling the examples...")
            random.shuffle(examples)

        if max_examples is not None:
            fprint(f"Detected max_examples={max_examples}, truncating the examples...")
            examples = examples[:max_examples]

        self.examples = examples
        return self.examples

    def prepare_input(self, instance, **kwargs):
        """
        Prepares a single data instance for the model. Must be implemented by subclasses.

        Args:
            instance (dict): A single data instance (e.g., a dictionary).
            **kwargs: Additional keyword arguments for tokenization.

        Returns:
            dict: A dictionary of tokenized inputs.
        """
        raise NotImplementedError(
            "The prepare_input() function should be implemented for your dataset."
        )

    @staticmethod
    def _download_dataset_from_hub(dataset_name, local_dir=None):
        """
        Downloads and extracts datasets from OmniGenBench Hub powered by HuggingFace.

        .. deprecated:: 0.3.23
            Use ``omnigenbench.src.utility.hub_utils.download_dataset`` instead.
            This method will be removed in version 0.4.0.

        Args:
            dataset_name (str): Name of the dataset to download.
            local_dir (str, optional): Directory to save the dataset. If None, saves to default location.

        Returns:
            str: Path to the downloaded dataset directory.
        """
        warnings.warn(
            "_download_dataset_from_hub() is deprecated. "
            "Use omnigenbench.src.utility.hub_utils.download_dataset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ...src.utility.hub_utils import download_dataset

        return download_dataset(dataset_name, cache_dir=local_dir, use_hf_api=True)

    @staticmethod
    def _download_dataset_from_huggingface(dataset_name, local_dir=None):
        """
        Downloads and extracts datasets from OmniGenBench Hub powered by HuggingFace.

        .. deprecated:: 0.3.0
            Use ``omnigenbench.src.utility.hub_utils.download_dataset`` instead.
            This method will be removed in version 0.4.0.

        Args:
            dataset_name (str): Name of the dataset to download.
            local_dir (str, optional): Directory to save the dataset. If None, saves to default location.

        Returns:
            str: Path to the downloaded dataset directory.
        """
        warnings.warn(
            "_download_dataset_from_huggingface() is deprecated. "
            "Use omnigenbench.src.utility.hub_utils.download_dataset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ...src.utility.hub_utils import download_dataset

        return download_dataset(dataset_name, cache_dir=local_dir, use_hf_api=True)

    def _preprocessing(self):
        """
        Performs preprocessing on the loaded examples.
        This method standardizes the 'sequence' field and adds secondary structure
        information if `structure_in` is True.
        """
        for idx, ex in enumerate(self.examples):
            if (
                "seq" in self.examples[idx]
            ):  # For the RNA or DNA stored in the "seq" field
                self.examples[idx]["sequence"] = self.examples[idx]["seq"]
                del self.examples[idx]["seq"]
            if (
                "text" in self.examples[idx]
            ):  # For the RNA or DNA stored in the "text" field
                self.examples[idx]["sequence"] = self.examples[idx]["text"]
                del self.examples[idx]["text"]

            if "sequence" not in self.examples[idx]:
                warnings.warn("The 'sequence' field is missing in the raw dataset.")
        if "sequence" in self.examples[0]:
            sequences = [ex["sequence"] for ex in self.examples]
            if self.structure_in:
                structures = self.rna2structure.fold(sequences)
                for idx, (sequence, structure) in enumerate(zip(sequences, structures)):
                    self.examples[idx][
                        "sequence"
                    ] = f"{sequence}{self.tokenizer.eos_token}{structure}"

    def _postprocessing(self):
        """
        Performs postprocessing on the tokenized data.
        This method standardizes the 'labels' field and prints the label distribution
        for classification tasks.
        """
        for idx, ex in enumerate(self.data):
            if "label" in self.data[idx]:
                self.data[idx]["labels"] = self.data[idx]["label"]
                # del self.data[idx]["label"]
            # assert (
            #         "labels" in self.data[idx]
            # ), "The 'labels' field is required in the tokenized dataset."

            if "labels" not in self.data[idx] or self.data[idx]["labels"] is None:
                self.data[idx]["labels"] = torch.tensor([-100])

        if self.data[0]["labels"].dim() == 0:
            self.print_label_distribution()

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single data sample at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            OmniDict: An `OmniDict` containing the data sample.
        """
        # convert the data item to a omnigenbench dict
        return OmniDict(self.data[idx])

    def sample(self, n=1) -> "OmniDataset":
        """
        Returns a random sample of n items from the dataset.

        Args:
            n (int): The number of samples to return.

        Returns:
            OmniDataset: A OmniDataset of data samples.
        """
        # Ensure n doesn't exceed dataset size
        n = min(n, len(self.data))

        # Randomly sample indices
        sampled_indices = random.sample(range(len(self.data)), n)

        # Create a new OmniDataset instance with None dataset_name_or_path to skip loading
        sampled_dataset = self.__class__(
            dataset_name_or_path=None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            label2id=self.label2id,
            shuffle=False,  # Don't shuffle sampled data
            structure_in=self.structure_in,
            drop_long_seq=self.drop_long_seq,
            force_padding=self.force_padding,
        )

        # Copy sampled data
        sampled_dataset.data = [self.data[i] for i in sampled_indices]
        sampled_dataset.examples = (
            [self.examples[i] for i in sampled_indices] if self.examples else []
        )

        # Copy metadata and other attributes
        sampled_dataset.metadata = self.metadata.copy()
        if hasattr(self, "id2label"):
            sampled_dataset.id2label = self.id2label
        if hasattr(self, "rna2structure"):
            sampled_dataset.rna2structure = self.rna2structure

        return sampled_dataset

    def get_column(self, column_name):
        """
        Returns all values for a specific column in the dataset.

        Args:
            column_name (str): The name of the column.

        Returns:
            list: A list of values from the specified column.
        """
        return [data_item[column_name] for data_item in self.data]

    def get_labels(self):
        """
        Returns the set of unique labels in the dataset.

        Returns:
            set: The set of unique labels.
        """
        return set(self.get_column("labels"))

    def get_inputs_length(self):
        """
        Calculates and returns statistics about sequence and label lengths.

        Returns:
            dict: A dictionary with length statistics (min, max, avg).
        """
        if hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.base_tokenizer.pad_token_id
        length = {}
        all_seq_lengths = [
            torch.sum(data_item["input_ids"] != pad_token_id) for data_item in self.data
        ]
        all_label_lengths = [
            data_item["labels"].shape[0] if data_item["labels"].shape else 1
            for data_item in self.data
        ]
        length["avg_seq_len"] = np.mean(all_seq_lengths)
        length["max_seq_len"] = np.max(all_seq_lengths)
        length["min_seq_len"] = np.min(all_seq_lengths)
        length["avg_label_len"] = np.mean(all_label_lengths)
        length["max_label_len"] = np.max(all_label_lengths)
        length["min_label_len"] = np.min(all_label_lengths)
        return length

    def _max_labels_length(self):
        """
        Returns the maximum length of labels in the dataset.

        Returns:
            int: The maximum length of labels.
        """
        if self.data[0]["labels"].dim() > 0:
            return max([len(ex["labels"]) for ex in self.data])
        else:
            return 1

    def __iter__(self):
        """
        Returns an iterator over the dataset.

        Returns:
            iterator: An iterator over the dataset.
        """
        for data_item in self.data:
            yield OmniDict(data_item)
