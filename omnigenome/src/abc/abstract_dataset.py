# -*- coding: utf-8 -*-
# file: abstract_dataset.py
# time: 14:13 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import os.path
import random
import warnings

import numpy as np
import torch
import tqdm
from transformers import BatchEncoding

from ..misc.utils import fprint, env_meta_info


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

        if data_source is not None and os.path.exists(data_source):
            fprint(f"Loading data from {data_source}...")
            self.load_data_source(data_source, **kwargs)

            for example in tqdm.tqdm(self.examples):
                self.data.append(self.prepare_input(example))

            if self.examples:
                self._post_processing()
                self._pad_and_truncate()
                self.data = covert_input_to_tensor(self.data)

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
        max_length = min(
            max(
                max(
                    torch.sum(data_item["input_ids"] != pad_token_id)
                    for data_item in self.data
                ),
                max(
                    torch.sum(data_item["labels"] != -100)
                    for data_item in self.data
                )
            ),
            self.max_length,
        )

        for data_item in self.data:
            for key, value in data_item.items():
                value = torch.tensor(np.array(value))
                dtype = value.dtype
                if isinstance(value, torch.Tensor) and value.dim() == 2:
                    padding_length = max_length - value.size(0)
                    if padding_length > 0:
                        if key == "input_ids":
                            if hasattr(self.tokenizer, "pad_token_id"):
                                _pad_value = self.tokenizer.pad_token_id * torch.ones(
                                    (padding_length, value.size(1))
                                )
                            else:
                                _pad_value = (
                                        self.tokenizer.base_tokenizer.pad_token_id
                                        * torch.ones((padding_length, value.size(1)))
                                )
                        elif key == "attention_mask":
                            _pad_value = torch.zeros((padding_length, value.size(1)))
                        elif "label" in key or "labels" in key:
                            _pad_value = -100 * torch.ones(
                                (padding_length, value.size(1))
                            )
                        else:
                            _pad_value = pad_value * torch.ones(
                                (padding_length, value.size(1))
                            )
                        data_item[key] = torch.cat([value, _pad_value], dim=0)
                    elif padding_length < 0:
                        data_item[key] = value[:max_length]
                    data_item[key] = data_item[key].to(dtype)

                elif isinstance(value, torch.Tensor) and value.dim() == 1:
                    padding_length = max_length - value.size(0)
                    if padding_length > 0:
                        if key == "input_ids":
                            if hasattr(self.tokenizer, "pad_token_id"):
                                _pad_value = self.tokenizer.pad_token_id * torch.ones(
                                    (padding_length,)
                                )
                            else:
                                _pad_value = (
                                        self.tokenizer.base_tokenizer.pad_token_id
                                        * torch.ones((padding_length,))
                                )
                        elif key == "attention_mask":
                            _pad_value = torch.zeros((padding_length,))
                        elif "label" in key or "labels" in key:
                            _pad_value = -100 * torch.ones((padding_length,))
                        else:
                            _pad_value = pad_value * torch.ones((padding_length,))
                        data_item[key] = torch.cat([value, _pad_value], dim=0)
                    elif padding_length < 0:
                        data_item[key] = value[:max_length]

                    data_item[key] = data_item[key].to(dtype)

    def load_data_source(self, data_source, **kwargs):
        examples = []
        max_examples = kwargs.get("max_examples", None)
        if data_source.endswith(".csv"):
            import pandas as pd

            df = pd.read_csv(data_source)
            for i in range(len(df)):
                examples.append(df.iloc[i].to_dict())
        elif data_source.endswith(".json"):
            import json

            with open(data_source, "r", encoding="utf8") as f:
                lines = f.readlines()
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

        if "shuffle" in kwargs and kwargs["shuffle"]:
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

    def _post_processing(self):
        for data in self.data:
            if "label" in data:
                data["labels"] = data["label"]
                del data["label"]

        print(self.get_sequence_length())

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

    def get_sequence_length(self):
        if hasattr(self.tokenizer, "pad_token_id"):
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = self.tokenizer.base_tokenizer.pad_token_id
        length = {}
        all_seq_lengths = [
            torch.sum(data_item["input_ids"] != pad_token_id) for data_item in self.data
        ]
        length["avg"] = np.mean(all_seq_lengths)
        length["max"] = np.max(all_seq_lengths)
        length["min"] = np.min(all_seq_lengths)
        return length

    def __iter__(self):
        for data_item in self.data:
            yield OmniGenomeDict(data_item)
