# -*- coding: utf-8 -*-
# file: bpe_tokenizer.py
# time: 18:32 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

from ..abc.abstract_tokenizer import OmniGenomeTokenizer
from transformers import AutoTokenizer


def is_bpe_tokenization(tokens, threshold=0.1):
    if not tokens:
        return False

    bpe_endings_count = sum(
        1
        for token in tokens
        if token.contains("##") or token.contains("@@") or token.contains("▁")
    )

    bpe_ratio = bpe_endings_count / len(tokens)

    return bpe_ratio >= threshold


class OmniBPETokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, **kwargs):
        super(OmniBPETokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        sequences = self.tokenize(sequence)

        if not is_bpe_tokenization(sequences):
            raise ValueError("The tokenizer seems not to be a BPE tokenizer.")

        tokenized_inputs = self.base_tokenizer(
            sequences,
            padding="do_not_pad",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs
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


if __name__ == "__main__":
    from transformers import AutoTokenizer

    RNA = "ACGUAGGUAUCGUAGA"
    base_tokenizer_name = "bert-base-cased"
    # base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    tokenizer = OmniBPETokenizer(base_tokenizer, max_length=512)
    tokens = tokenizer.tokenize(RNA)
    print(tokens)
    tokenized_inputs = tokenizer(RNA)
    print(tokenized_inputs)
