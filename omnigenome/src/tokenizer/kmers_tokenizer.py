# -*- coding: utf-8 -*-
# file: kmers_tokenizer.py
# time: 18:31 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import warnings

from ..abc.abstract_tokenizer import OmniGenomeTokenizer
from transformers import AutoTokenizer


class OmniKmersTokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, k=3, overlap=0, max_length=512, **kwargs):
        super(OmniKmersTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.k = k
        self.overlap = overlap
        self.max_length = max_length
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        if self.u2t:
            sequence = sequence.replace("U", "T")
        if self.add_whitespace:
            sequence = " ".join(list(sequence))
        sequence_tokens = self.tokenize(sequence)
        tokenized_inputs = {
            "input_ids": [],
            "attention_mask": [],
        }
        bos_id, eos_id = self.base_tokenizer("")["input_ids"]

        for tokens in sequence_tokens:
            tokenized_inputs["input_ids"].append(
                [bos_id]
                + self.base_tokenizer.convert_tokens_to_ids(
                    tokens[: kwargs.get("max_length", self.max_length) - 2]
                )
                + [eos_id]
            )
            tokenized_inputs["attention_mask"].append(
                [1] * len(tokenized_inputs["input_ids"][-1])
            )

        for i, ids in enumerate(tokenized_inputs["input_ids"]):
            if ids.count(self.base_tokenizer.unk_token_id) / len(ids) > 0.1:
                warnings.warn(
                    f"Unknown tokens are more than 10% in the {i}th sequence, please check the tokenization process."
                )
        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding="max_length",
            max_length=self.max_length,
            pad_to_multiple_of=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        self = OmniKmersTokenizer(
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
            tokens = []
            for j in range(0, len(sequences[i]), self.k - self.overlap):
                tokens.append(sequences[i][j : j + self.k])

            sequence_tokens.append(tokens)

        return sequence_tokens

    def encode(self, input_ids, **kwargs):
        return self.base_tokenizer.encode(input_ids, **kwargs)

    def decode(self, input_ids, **kwargs):
        return self.base_tokenizer.decode(input_ids, **kwargs)

    def encode_plus(self, sequence, **kwargs):
        raise NotImplementedError("The encode_plus() function is not implemented yet.")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # RNA = "ACGUAGGUAUCGUAGA"
    # # base_tokenizer_name = 'bert-base-cased'
    # base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    # base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    # tokenizer = KmersTokenizer(base_tokenizer)
    # tokens = tokenizer.tokenize(RNA)
    # print(tokens)
    # tokenized_inputs = tokenizer(RNA)
    # print(tokenized_inputs)

    RNA = "ACGUAGGUAUCGUAGA"
    # base_tokenizer_name = 'bert-base-cased'
    base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    tokenizer = OmniKmersTokenizer(base_tokenizer, k=4, overlap=2, max_length=512)
    tokens = tokenizer.tokenize(RNA)
    print(tokens)
    tokenized_inputs = tokenizer(RNA)
    print(tokenized_inputs)
