# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import random
import warnings
from collections import Counter

import numpy as np
import torch
import tqdm

from transformers import BatchEncoding

from ..misc.utils import fprint, env_meta_info, RNA2StructureCache


def load_omnigenome_dataset(benchmark=None, dataset='', **kwargs):
    """
    Load the OmniGenome dataset for a specific benchmark and dataset name.
    This function will search for the dataset files in the specified benchmark directory.
    :param benchmark: The benchmark name, such as "BEACON", "GUE", "RGB", etc.
    :param dataset: The dataset name, such as "train", "test", "valid", etc.
    :param kwargs: Additional keyword arguments for the dataset.
    :return: A collection of instances without tokenization.
    """
    import os
    from findfile import find_file, find_files
    from ... import download_benchmark

    if not benchmark:
        assert dataset, "Either benchmark or dataset (name or path) must be provided."
        benchmark = dataset

    if not os.path.exists(benchmark):
        fprint(
            "Benchmark:",
            benchmark,
            "does not exist. Search online for available benchmarks.",
        )
        benchmark_root = download_benchmark(benchmark)
    else:
        benchmark_root = benchmark

    data_cfg_file = find_file(benchmark_root, ['config', dataset])
    data_files = find_files(
        os.path.dirname(data_cfg_file),
        or_key=['train', 'dev', 'test', 'valid', dataset],
        exclude_key=['.py', '__pycache__', '.ipynb_checkpoints'],
    )
    examples = {}
    max_examples = kwargs.get("max_examples", None)

    for data_source in data_files:
        split = []

        if data_source.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(data_source)
            for i in range(len(df)):
                split.append(df.iloc[i].to_dict())
        elif data_source.endswith(".json"):
            import json

            try:
                with open(data_source, "r", encoding="utf8") as f:
                    split = json.load(f)
            except:
                with open(data_source, "r", encoding="utf8") as f:
                    lines = f.readlines()  # Assume the data is a list of examples
                for i in range(len(lines)):
                    lines[i] = json.loads(lines[i])
                for line in lines:
                    split.append(line)
        elif data_source.endswith(".parquet"):
            import pandas as pd

            df = pd.read_parquet(data_source)
            for i in range(len(df)):
                split.append(df.iloc[i].to_dict())
        elif data_source.endswith(".txt") or data_source.endswith(".dat"):
            with open(data_source, "r", encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                split.append({"text": line.strip()})
        else:
            raise Exception(f"Unknown file format of {data_source}.")

        fprint(f"Loaded {len(split)} examples from {data_source}")

        if kwargs.get("shuffle", True) is True:
            fprint("Detected shuffle=True, shuffling the examples...")
            random.shuffle(split)

        if max_examples is not None:
            fprint(f"Detected max_examples={max_examples}, truncating the examples...")
            split = split[:max_examples]

        examples[os.path.basename(data_source)] = split

    return examples


def covert_input_to_tensor(data):
    """
    Convert the data in the dataset to PyTorch tensors.
    :param data: A list of dictionaries, where each dictionary represents a data sample.
    :return: The data in the dataset as PyTorch tensors.
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


class OmniGenomeDict(dict):
    def __init__(self, *args, **kwargs):
        super(OmniGenomeDict, self).__init__(*args, **kwargs)

    def to(self, device):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)
        return self


class OmniGenomeDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, tokenizer, max_length=None, **kwargs):
        super(OmniGenomeDataset, self).__init__()
        self.metadata = env_meta_info()
        self.tokenizer = tokenizer
        self.label2id = kwargs.get("label2id", None)
        self.shuffle = kwargs.get("shuffle", True)
        self.structure_in = kwargs.get("structure_in", False)
        self.drop_long_seq = kwargs.get("drop_long_seq", False)
        if self.structure_in and not hasattr(self, "rna2structure"):
            self.rna2structure = RNA2StructureCache()

        if self.label2id is not None:
            self.id2label = {v: k for k, v in self.label2id.items()}

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
            raise ValueError("max_length must be provided in the dataset or tokenizer.")

        self.tokenizer.max_length = self.max_length
        self.examples = []
        self.data = []

        if data_source is not None:
            fprint(f"Loading data from {data_source}...")
            self.load_data_source(data_source, **kwargs)
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

            if self.examples:
                self.data = covert_input_to_tensor(self.data)
                self._pad_and_truncate()
                fprint(self.get_inputs_length())
                fprint(f"Preview of the first two samples in the dataset:")
                for sample in self.data[:2]:
                    fprint(sample)

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
        for data_item in self.data:
            for key, value in data_item.items():
                if isinstance(value, torch.Tensor):
                    data_item[key] = value.to(device)
        return self

    def _pad_and_truncate(self, pad_value=0):
        if hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.base_tokenizer.pad_token_id

        # 计算输入和标签的最大长度
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

        # 确定初始max_length，不超过self.max_length
        original_max_length = max(max_input_length, max_label_length)
        original_max_length = min(original_max_length, self.max_length)

        # 调整到不超过self.max_length的最大的8的倍数
        remainder = original_max_length % 8
        if remainder != 0:
            adjusted_max_length = original_max_length + (8 - remainder)
            adjusted_max_length = min(adjusted_max_length, self.max_length)
        else:
            adjusted_max_length = original_max_length
        max_length = adjusted_max_length

        # 处理标签的特殊情况（修复错误的关键部分）
        first_labels = self.data[0]["labels"]

        label_shape = first_labels.shape
        if len(label_shape) >= 1:
            label_padding_length = max(max_length, self.data[0]["labels"].shape[0])
            label_padding_length = min(label_padding_length, max_length)
            max_length = max(max_length, label_padding_length)
        else:
            label_padding_length = 0

        fprint(
            f"Max sequence length updated -> Reset max_length={max_length},"
            f" label_padding_length={label_padding_length}"
        )

        for data_item in self.data:
            for key, value in data_item.items():
                # 确保转换为Tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                dtype = value.dtype
                if "label" in key and (
                    value.dtype == torch.int16 or value.dtype == torch.int32
                ):
                    data_item[key] = value.long()
                # 确定填充长度
                if "label" in key:
                    if value.ndim == 0:  # 处理标量标签
                        padding_length = 0
                    else:
                        padding_length = label_padding_length - value.size(0)
                else:
                    padding_length = max_length - value.size(0)

                # 处理填充或截断
                if padding_length > 0:
                    # 确定填充值
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

                    # 构建填充张量
                    if value.ndim == 2:
                        pad_shape = (padding_length, value.size(1))
                    else:
                        pad_shape = (padding_length,)
                    pad_tensor = torch.full(pad_shape, _pad_value, dtype=dtype)
                    data_item[key] = torch.cat([value, pad_tensor], dim=0)
                elif padding_length < 0:
                    data_item[key] = value[:max_length]

                # 确保数据类型正确
                data_item[key] = data_item[key].to(dtype)

        return self.data

    def load_data_source(self, data_source, **kwargs):
        examples = []
        max_examples = kwargs.get("max_examples", None)
        if not isinstance(data_source, list):
            data_source = [data_source]

        for data_source in data_source:
            if data_source.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(data_source)
                for i in range(len(df)):
                    examples.append(df.iloc[i].to_dict())
            elif data_source.endswith(".json"):
                import json

                try:
                    with open(data_source, "r", encoding="utf8") as f:
                        examples = json.load(f)
                except:
                    with open(data_source, "r", encoding="utf8") as f:
                        lines = f.readlines()  # Assume the data is a list of examples
                    for i in range(len(lines)):
                        lines[i] = json.loads(lines[i])
                    for line in lines:
                        examples.append(line)
            elif data_source.endswith(".parquet"):
                import pandas as pd

                df = pd.read_parquet(data_source)
                for i in range(len(df)):
                    examples.append(df.iloc[i].to_dict())
            elif data_source.endswith(".txt") or data_source.endswith(".dat"):
                with open(data_source, "r", encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    examples.append({"text": line.strip()})
            else:
                raise Exception("Unknown file format.")

        fprint(f"Loaded {len(examples)} examples from {data_source}")

        if self.shuffle is True:
            fprint("Detected shuffle=True, shuffling the examples...")
            random.shuffle(examples)

        if max_examples is not None:
            fprint(f"Detected max_examples={max_examples}, truncating the examples...")
            examples = examples[:max_examples]

        self.examples = examples
        return examples

    def prepare_input(self, instance, **kwargs):
        raise NotImplementedError(
            "The prepare_input() function should be implemented for your dataset."
        )

    def _preprocessing(self):
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
        for idx, ex in enumerate(self.data):
            if "label" in self.data[idx]:
                self.data[idx]["labels"] = self.data[idx]["label"]
                # del self.data[idx]["label"]
            # assert (
            #         "labels" in self.data[idx]
            # ), "The 'labels' field is required in the tokenized dataset."

            if "labels" not in self.data[idx].data or self.data[idx]["labels"] is None:
                self.data[idx]["labels"] = torch.tensor([-100])

        if self.data[0]["labels"].dim() == 0:
            self.print_label_distribution()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert the data item to a omnigenome dict
        return OmniGenomeDict(self.data[idx])

    def sample(self, n=1):
        return random.sample(self.data, n)

    def get_column(self, column_name):
        return [data_item[column_name] for data_item in self.data]

    def get_labels(self):
        return set(self.get_column("labels"))

    def get_inputs_length(self):
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
        if self.data[0]["labels"].dim() > 0:
            return max([len(ex["labels"]) for ex in self.data])
        else:
            return 1

    def __iter__(self):
        for data_item in self.data:
            yield OmniGenomeDict(data_item)
