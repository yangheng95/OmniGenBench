# -*- coding: utf-8 -*-
# file: single_nucleotide_tokenizer.py
# time: 18:05 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import pickle
import warnings

from ..abc.abstract_tokenizer import OmniGenomeTokenizer


class OmniSingleNucleotideTokenizer(OmniGenomeTokenizer):
    def __init__(self, base_tokenizer=None, **kwargs):
        super(OmniSingleNucleotideTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = "SingleNucleotideTokenizer"

    def __call__(self, sequence, **kwargs):
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
                    tokens[: self.max_length - 2]
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
    base_tokenizer_name = "bert-base-cased"
    # base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    tokenizer = OmniSingleNucleotideTokenizer(base_tokenizer, max_length=512)
    tokens = tokenizer.tokenize(RNA)
    print(tokens)
    tokenized_inputs = tokenizer(RNA)
    print(tokenized_inputs)
    pickle.dump(tokenizer, open("tokenizer.og.pkl", "wb"))
    tokenizer = pickle.load(open("tokenizer.og.pkl", "rb"))
    tokenized_inputs = tokenizer(RNA)
