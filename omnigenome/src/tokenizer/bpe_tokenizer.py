# -*- coding: utf-8 -*-
# file: bpe_tokenizer.py
# time: 18:32 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import numpy as np
from ..abc.abstract_tokenizer import OmniGenomeTokenizer
from transformers import AutoTokenizer


def is_bpe_tokenization(tokens, threshold=0.1):
    if not tokens:
        return False

    # bpe_endings_count = sum(
    #     1
    #     for token in tokens
    #     if token.startswith("##") or token.startswith("@@") or token.startswith("▁")
    # )
    # bpe_ratio = bpe_endings_count / len(tokens)

    rmse = np.mean([len(token) ** 2 for token in tokens]) ** 0.5

    return rmse >= threshold


class OmniBPETokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, **kwargs):
        super(OmniBPETokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        if self.u2t:
            sequence = sequence.replace("U", "T")
        if self.add_whitespace:
            sequence = " ".join(list(sequence))

        sequences = self.tokenize(sequence)[
            : min(self.max_length, kwargs.get("max_length", 512)) - 2
        ]

        if not is_bpe_tokenization(sequences):
            raise ValueError("The tokenizer seems not to be a BPE tokenizer.")
        tokenized_inputs = dict()
        tokenized_inputs["input_ids"] = self.base_tokenizer.convert_tokens_to_ids(
            sequences
        )
        tokenized_inputs["attention_mask"] = [1] * len(tokenized_inputs["input_ids"])

        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding=kwargs.get("padding", "max_length"),
            max_length=min(self.max_length, kwargs.get("max_length", 512)),
            return_attention_mask=kwargs.get("return_attention_mask", True),
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        self = OmniBPETokenizer(
            AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        return self.base_tokenizer.tokenize(sequence)

    def encode(self, sequence, **kwargs):
        assert hasattr(
            self.base_tokenizer, "bpe"
        ), "The base tokenizer must be a BPE tokenizer."
        return self.base_tokenizer.encode(sequence, **kwargs)

    def decode(self, sequence, **kwargs):
        assert hasattr(
            self.base_tokenizer, "bpe"
        ), "The base tokenizer must be a BPE tokenizer."
        return self.base_tokenizer.decode(sequence, **kwargs)
