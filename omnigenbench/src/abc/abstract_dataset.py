# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
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
    A unified interface for genomic datasets in the OmniGenome
    framework. It handles data loading, preprocessing, tokenization, and provides
    a PyTorch-compatible dataset interface.

    The class supports various data formats and can handle different types of
    genomic tasks including classification, regression, and token-level tasks.

    Attributes:
        tokenizer: The tokenizer to use for processing sequences.
        max_length (int): The maximum sequence length for tokenization.
        label2id (dict): Mapping from labels to integer IDs.
        id2label (dict): Mapping from integer IDs to labels.
        shuffle (bool): Whether to shuffle the data.
        structure_in (bool): Whether to include secondary structure information.
        drop_long_seq (bool): Whether to drop sequences longer than max_length.
        metadata (dict): Metadata about the dataset including version info.
        rna2structure (RNA2StructureCache): Cache for RNA structure predictions.
    """

    def __init__(self, dataset_name_or_path, tokenizer, max_length=None, **kwargs):
        """
        Initializes the dataset.

        Args:
            dataset_name_or_path (str or list): Path to the data file or a list of paths.
            tokenizer: The tokenizer to use for processing sequences.
            max_length (int, optional): The maximum sequence length.
            **kwargs: Additional keyword arguments.
                - label2id (dict): A mapping from labels to integer IDs.
                - shuffle (bool): Whether to shuffle the data. Defaults to True.
                - structure_in (bool): Whether to include secondary structure
                  information. Defaults to False.
                - drop_long_seq (bool): Whether to drop sequences longer than
                  max_length. Defaults to False.
                - dataset_url (str): URL to download dataset if not found locally.
                - cache_dir (str): Directory to cache downloaded datasets.

        Example:
            >>> # Initialize with a single data file
            >>> dataset = OmniDataset("data.json", tokenizer, max_length=512)

            >>> # Initialize with label mapping
            >>> dataset = OmniDataset("data.json", tokenizer,
            ...                       label2id={"A": 0, "B": 1})

            >>> # Initialize with automatic dataset download
            >>> dataset = OmniDataset("data.csv", tokenizer,
            ...                       dataset_url="https://example.com/data.zip")
        """
        super(OmniDataset, self).__init__()
        self.metadata = env_meta_info()
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
            self.load_data_source(dataset_name_or_path, **kwargs)
            self._preprocessing()

            for example in tqdm.tqdm(self.examples):
                if hasattr(self.tokenizer, "max_length"):
                    self.tokenizer.max_length = self.max_length
                else:
                    self.tokenizer.base_tokenizer.max_length = self.max_length

                import inspect

                new_args = {}
                tokenization_args = inspect.getfullargspec(self.tokenizer.encode).args
                for key in kwargs:
                    if key in tokenization_args:
                        new_args[key] = kwargs[key]
                prepared_input = self.prepare_input(example, **new_args)

                # Squeeze the batch dimension if it exists
                for key, value in prepared_input.items():
                    prepared_input[key] = value.squeeze(0)

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
                    self.data.append(prepared_input)

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
            cls._download_dataset_from_hub(dataset_name_or_path, cache_dir)
        else:
            fprint(
                f"Loading dataset from local path: {dataset_name_or_path or cache_dir}"
            )

        # Create datasets for each split
        datasets = {"train": None, "valid": None, "test": None}
        for split in splits:
            if not cache_dir:
                if is_local and os.path.exists(dataset_name_or_path):
                    # If dataset_name_or_path itself is a local directory
                    cache_dir = dataset_name_or_path
                else:
                    # Use default cache directory
                    cache_dir = os.getcwd()

            data_source = findfile.find_files(
                cache_dir, [split], exclude_key=[".ipynb", ".py", "md", "txt"]
            )
            if not data_source:
                fprint(
                    f"Warning: No data files found for split '{split}' in {cache_dir}. Skipping this split."
                )
                continue
            else:
                fprint(f"Load data files for split '{split}': {data_source}")

            datasets[split] = cls(
                dataset_name_or_path=data_source,
                tokenizer=tokenizer,
                max_length=max_length,
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

        .. deprecated::
            `from_huggingface` is deprecated and will be removed in a future version.
            Use `from_huggingface` instead, which supports both HuggingFace Hub and local data sources.

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
            "from_huggingface() is deprecated and will be removed in a future version. "
            "Please use from_huggingface() instead, which supports both HuggingFace Hub and local data sources.",
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
            label_padding_length = min(max_label_length, max_length)
            max_length = max(max_length, label_padding_length)
        else:
            label_padding_length = 0

        fprint(
            f"Max sequence length updated -> Reset max_length={max_length},"
            f" label_padding_length={label_padding_length}"
        )

        for data_item in self.data:
            for key, value in data_item.items():

                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                dtype = value.dtype
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

    def load_data_source(self, data_source, **kwargs):
        """
        Loads data from a file or list of files.

        Args:
            data_source (str or list): Path to the data file or a list of paths.
            **kwargs: Additional keyword arguments, e.g., `max_examples`.

        Returns:
            list: A list of examples.
        """
        examples = []
        max_examples = kwargs.get("max_examples", None)
        columns = kwargs.get("select_columns", None)
        if not isinstance(data_source, list):
            data_source = [data_source]

        for data_source in data_source:
            _examples = []

            if data_source.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(data_source)
                for i in range(len(df)):
                    _examples.append(df.iloc[i].to_dict())
            elif data_source.endswith(".json") or data_source.endswith(".jsonl"):
                import json

                try:
                    with open(data_source, "r", encoding="utf8") as f:
                        _examples = json.load(f)
                except:
                    with open(data_source, "r", encoding="utf8") as f:
                        lines = f.readlines()  # Assume the data is a list of examples
                    for i in range(len(lines)):
                        lines[i] = json.loads(lines[i])
                    for line in lines:
                        _examples.append(line)
            elif data_source.endswith(".parquet"):
                import pandas as pd

                df = pd.read_parquet(data_source)
                for i in range(len(df)):
                    _examples.append(df.iloc[i].to_dict())
            elif data_source.endswith(".npy") or data_source.endswith(".npz"):
                import numpy as np

                if data_source.endswith(".npy"):
                    data = np.load(data_source, allow_pickle=True)
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
                elif data_source.endswith(".npz"):
                    data = np.load(data_source, allow_pickle=True)
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
            elif data_source.endswith(
                (".fasta", ".fa", ".fna", ".ffn", ".faa", ".frn")
            ):
                try:
                    from Bio import SeqIO
                except ImportError:
                    raise ImportError(
                        "Biopython is required for FASTA parsing. Please install with 'pip install biopython'."
                    )
                for record in SeqIO.parse(data_source, "fasta"):
                    _examples.append(
                        {
                            "id": record.id,
                            "sequence": str(record.seq),
                            "description": record.description,
                        }
                    )
            elif data_source.endswith((".fastq", ".fq")):
                try:
                    from Bio import SeqIO
                except ImportError:
                    raise ImportError(
                        "Biopython is required for FASTQ parsing. Please install with 'pip install biopython'."
                    )
                for record in SeqIO.parse(data_source, "fastq"):
                    _examples.append(
                        {
                            "id": record.id,
                            "sequence": str(record.seq),
                            "quality": record.letter_annotations.get(
                                "phred_quality", []
                            ),
                        }
                    )
            elif data_source.endswith(".bed"):
                import pandas as pd

                df = pd.read_csv(data_source, sep="\t", comment="#")
            else:
                raise Exception("Unknown file format.")

            if columns := kwargs.get("select_columns", None):
                fprint(f"Selecting columns: {columns}")
                filtered_examples = []
                for ex in _examples:
                    filtered_ex = {col: ex[col] for col in columns if col in ex}
                    filtered_examples.append(filtered_ex)
                _examples = filtered_examples

            examples.extend(_examples)
            del _examples

            fprint(
                f"Reading from {data_source}, Loaded {len(examples)} examples so far..."
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

        This method supports downloading datasets from the OmniGenBench Hub on HuggingFace.

        Args:
            dataset_name (str): Name of the dataset to download.
            local_dir (str, optional): Directory to save the dataset. If None, saves to default location.
        """
        if local_dir is None:
            local_dir = os.path.join(
                os.getcwd(), f"__OMNIGENOME_DATA__/datasets/{dataset_name}"
            )
        else:
            local_dir = os.path.abspath(local_dir)

        url_to_download = f"https://huggingface.co/datasets/yangheng/OmniGenBench_Hub/resolve/main/{dataset_name}.zip"
        zip_path = os.path.join(local_dir, f"{dataset_name}.zip")

        if not os.path.exists(local_dir):
            if not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            fprint(f"Downloading dataset from {url_to_download}...")
            response = requests.get(url_to_download, stream=True)
            response.raise_for_status()

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            fprint(f"Downloaded {zip_path}")

        # Unzip the dataset
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(local_dir)
            os.remove(zip_path)
        else:
            fprint(
                f"Dataset already downloaded and extracted at {local_dir}."
                f"If you want to re-download, please delete the existing directory."
            )

    @staticmethod
    def _download_dataset_from_huggingface(dataset_name, local_dir=None):
        """
        Downloads and extracts datasets from OmniGenBench Hub on powered by HuggingFace.

        .. deprecated::
            Use `_download_dataset_from_huggingface` instead.
        """
        warnings.warn(
            "_download_dataset_from_huggingface() is deprecated. "
            "Use _download_dataset_from_huggingface() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OmniDataset._download_dataset_from_hub(dataset_name, local_dir)

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
