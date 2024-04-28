# -*- coding: utf-8 -*-
# file: single_nucleotide_tokenizer.py
# time: 18:05 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import warnings

from ..abc.abstract_tokenizer import OmniGenomeTokenizer
from transformers import AutoTokenizer


class OmniSingleNucleotideTokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, **kwargs):
        super(OmniSingleNucleotideTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        if self.u2t:
            sequence = "".join([seq.replace("U", "T").upper() for seq in sequence])
        if self.t2u:
            sequence = "".join([seq.replace("T", "U").upper() for seq in sequence])
        if self.add_whitespace:
            sequence = " ".join(list(sequence))
        sequence_tokens = self.tokenize(sequence)[
            : kwargs.get("max_length", self.max_length) - 2
        ]
        tokenized_inputs = {
            "input_ids": [],
            "attention_mask": [],
        }
        bos_id = (
            self.base_tokenizer.bos_token_id
            if self.base_tokenizer.bos_token_id is not None
            else self.base_tokenizer.cls_token_id
        )
        eos_id = (
            self.base_tokenizer.eos_token_id
            if self.base_tokenizer.eos_token_id is not None
            else self.base_tokenizer.sep_token_id
        )
        for tokens in sequence_tokens:
            tokenized_inputs["input_ids"].append(
                [bos_id] + self.base_tokenizer.convert_tokens_to_ids(tokens) + [eos_id]
            )
            tokenized_inputs["attention_mask"].append(
                [1] * len(tokenized_inputs["input_ids"][-1])
            )

        for i, ids in enumerate(tokenized_inputs["input_ids"]):
            if ids.count(self.base_tokenizer.unk_token_id) / len(ids) > 0.1:
                warnings.warn(
                    f"Unknown tokens are more than "
                    f"{ids.count(self.base_tokenizer.unk_token_id) / len(ids)}% in the {i}-th sequence, "
                    f"please check the tokenization process."
                )
        max_length = max(len(ids) for ids in tokenized_inputs["input_ids"])
        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding=kwargs.get("padding", "max_length"),
            max_length=min(max_length, kwargs.get("max_length", 512)),
            return_attention_mask=kwargs.get("return_attention_mask", True),
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        self = OmniSingleNucleotideTokenizer(
            AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        if isinstance(sequence, str):
            sequences = [sequence]
        else:
            sequences = sequence

        sequence_tokens = []
        for i in range(len(sequences)):
            sequence_tokens.append(list(sequences[i]))

        return sequence_tokens

    def encode(self, sequence, **kwargs):
        return self.base_tokenizer.encode(sequence, **kwargs)

    def decode(self, sequence, **kwargs):
        return self.base_tokenizer.decode(sequence, **kwargs)

    def encode_plus(self, sequence, **kwargs):
        return self.base_tokenizer.encode_plus(sequence, **kwargs)
