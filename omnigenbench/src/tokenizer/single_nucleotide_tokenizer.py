# -*- coding: utf-8 -*-
# file: single_nucleotide_tokenizer.py
# time: 18:05 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import warnings

from transformers import AutoTokenizer

from ..abc.abstract_tokenizer import OmniTokenizer

warnings.filterwarnings("once")


class OmniSingleNucleotideTokenizer(OmniTokenizer):
    """
    Tokenizer for single nucleotide tokenization in genomics.

    This tokenizer converts genomic sequences into individual nucleotide tokens,
    where each nucleotide (A, T, C, G, U) becomes a separate token. It's designed
    for genomic sequence processing where fine-grained nucleotide-level analysis
    is required.

    The tokenizer supports various preprocessing options including U/T conversion
    and whitespace addition between nucleotides. It also handles special tokens
    like BOS (beginning of sequence) and EOS (end of sequence) tokens.

    Attributes:
        u2t (bool): Whether to convert 'U' to 'T'.
        t2u (bool): Whether to convert 'T' to 'U'.
        add_whitespace (bool): Whether to add whitespace between nucleotides.
    """

    def __init__(self, base_tokenizer=None, **kwargs):
        """
        Initializes the single nucleotide tokenizer.

        Args:
            base_tokenizer: The underlying Hugging Face tokenizer.
            **kwargs: Additional keyword arguments passed to the parent class.

        Example:
            >>> from transformers import AutoTokenizer
            >>> base_tokenizer = AutoTokenizer.from_pretrained("model_name")
            >>> tokenizer = OmniSingleNucleotideTokenizer(base_tokenizer)
        """
        super(OmniSingleNucleotideTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        """
        Tokenizes sequences using single nucleotide tokenization.

        This method converts genomic sequences into tokenized inputs suitable
        for model training and inference. It handles sequence preprocessing,
        tokenization, and padding/truncation.

        Args:
            sequence (str or list): A single sequence or list of sequences to tokenize.
            **kwargs: Additional arguments for tokenization:
                - max_length (int): Maximum sequence length.
                - padding (str): Padding strategy.
                - truncation (bool): Whether to truncate sequences.
                - warnings (bool): Whether to show warnings for unknown tokens.

        Returns:
            dict: A dictionary containing tokenized inputs:
                - input_ids: Token IDs for the sequences
                - attention_mask: Attention mask for the sequences

        Example:
            >>> # Tokenize a single sequence
            >>> inputs = tokenizer("ATCGATCG")
            >>> print(inputs['input_ids'].shape)  # torch.Size([1, seq_len])

            >>> # Tokenize multiple sequences
            >>> inputs = tokenizer(["ATCGATCG", "GCTAGCTA"])
            >>> print(inputs['input_ids'].shape)  # torch.Size([2, seq_len])
        """
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

        if kwargs.get("warnings", True):
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
        """
        Loads a single nucleotide tokenizer from a pre-trained model.

        This method creates a single nucleotide tokenizer wrapper around
        a Hugging Face tokenizer loaded from a pre-trained model.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            **kwargs: Additional arguments for the tokenizer.

        Returns:
            OmniSingleNucleotideTokenizer: An instance of the tokenizer.

        Example:
            >>> tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
        """
        self = OmniSingleNucleotideTokenizer(
            AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        """
        Converts a sequence into a list of individual nucleotide tokens.

        This method tokenizes genomic sequences by treating each nucleotide
        as a separate token. It handles both single sequences and lists of sequences.

        Args:
            sequence (str or list): A single sequence or list of sequences to tokenize.
            **kwargs: Additional arguments (not used in this implementation).

        Returns:
            list: A list of token lists, where each inner list contains
                  individual nucleotide tokens.

        Example:
            >>> # Tokenize a single sequence
            >>> tokens = tokenizer.tokenize("ATCGATCG")
            >>> print(tokens)  # [['A', 'T', 'C', 'G', 'A', 'T', 'C', 'G']]

            >>> # Tokenize multiple sequences
            >>> tokens = tokenizer.tokenize(["ATCGATCG", "GCTAGCTA"])
            >>> print(tokens)  # [['A', 'T', 'C', 'G', ...], ['G', 'C', 'T', 'A', ...]]
        """
        if isinstance(sequence, str):
            sequences = [sequence]
        else:
            sequences = sequence

        sequence_tokens = []
        for i in range(len(sequences)):
            sequence_tokens.append(list(sequences[i]))

        return sequence_tokens

    def encode(self, sequence, **kwargs):
        """
        Converts a sequence into a list of token IDs.

        This method encodes genomic sequences into token IDs using the
        underlying base tokenizer.

        Args:
            sequence (str): The input sequence to encode.
            **kwargs: Additional arguments for encoding.

        Returns:
            list: A list of token IDs.

        Example:
            >>> token_ids = tokenizer.encode("ATCGATCG")
            >>> print(token_ids)  # [1, 2, 3, 4, 1, 2, 3, 4]
        """
        return self.base_tokenizer.encode(sequence, **kwargs)

    def decode(self, sequence, **kwargs):
        """
        Converts a list of token IDs back into a sequence.

        This method decodes token IDs back into genomic sequences using
        the underlying base tokenizer.

        Args:
            sequence (list): A list of token IDs.
            **kwargs: Additional arguments for decoding.

        Returns:
            str: The decoded sequence.

        Example:
            >>> sequence = tokenizer.decode([1, 2, 3, 4])
            >>> print(sequence)  # "ATCG"
        """
        return self.base_tokenizer.decode(sequence, **kwargs)

    def encode_plus(self, sequence, **kwargs):
        """
        Encodes a sequence with additional information.

        This method provides enhanced encoding with additional information
        like attention masks and token type IDs.

        Args:
            sequence (str): The input sequence to encode.
            **kwargs: Additional arguments for encoding.

        Returns:
            dict: A dictionary containing encoded information.

        Example:
            >>> encoded = tokenizer.encode_plus("ATCGATCG")
            >>> print(encoded.keys())  # dict_keys(['input_ids', 'attention_mask'])
        """
        return self.base_tokenizer.encode_plus(sequence, **kwargs)
