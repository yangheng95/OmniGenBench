# -*- coding: utf-8 -*-
# file: bpe_tokenizer.py
# time: 18:32 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import numpy as np
import warnings

from ..abc.abstract_tokenizer import OmniTokenizer

warnings.filterwarnings("once")


def is_bpe_tokenization(tokens, threshold=0.1):
    """
    Check if the tokenization is BPE-based by analyzing token characteristics.

    This function examines the tokens to determine if they follow BPE tokenization
    patterns by analyzing token length distributions and special token patterns.

    Args:
        tokens (list): List of tokens to analyze
        threshold (float, optional): Threshold for determining BPE tokenization. Defaults to 0.1

    Returns:
        bool: True if tokens appear to be BPE-based, False otherwise

    Example:
        >>> tokens = ["▁hello", "▁world", "▁how", "▁are", "▁you"]
        >>> is_bpe = is_bpe_tokenization(tokens)
        >>> print(is_bpe)
        True
    """
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


class OmniBPETokenizer(OmniTokenizer):
    """
    A Byte Pair Encoding (BPE) tokenizer for genomic sequences.

    This tokenizer uses BPE tokenization for genomic sequences and provides
    validation to ensure the base tokenizer is BPE-based. It supports sequence
    preprocessing and handles various input formats.

    Attributes:
        base_tokenizer: The underlying BPE tokenizer
        metadata: Dictionary containing tokenizer metadata

    Example:
        >>> from omnigenbench import OmniBPETokenizer
        >>> from transformers import AutoTokenizer
        >>> base_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        >>> tokenizer = OmniBPETokenizer(base_tokenizer)
        >>> sequence = "ACGUAGGUAUCGUAGA"
        >>> tokens = tokenizer.tokenize(sequence)
        >>> print(tokens[:5])
        ['▁A', 'C', 'G', 'U', 'A']
    """

    def __init__(self, base_tokenizer=None, **kwargs):
        """
        Initialize the OmniBPETokenizer.

        Args:
            base_tokenizer: The base BPE tokenizer
            **kwargs: Additional keyword arguments passed to parent class
        """
        super(OmniBPETokenizer, self).__init__(base_tokenizer, **kwargs)
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        """
        Tokenize a sequence using BPE tokenization.

        This method processes the input sequence using BPE tokenization,
        handles sequence preprocessing (U/T conversion, whitespace addition),
        and validates that the tokenization is BPE-based.

        Args:
            sequence (str): Input sequence to tokenize
            **kwargs: Additional keyword arguments including max_length

        Returns:
            dict: Dictionary containing tokenized inputs with keys 'input_ids' and 'attention_mask'

        Raises:
            ValueError: If the tokenizer is not BPE-based

        Example:
            >>> sequence = "ACGUAGGUAUCGUAGA"
            >>> tokenized = tokenizer(sequence)
            >>> print(tokenized['input_ids'].shape)
            torch.Size([1, 17])
        """
        if self.u2t:
            sequence = sequence.replace("U", "T")
        if self.add_whitespace:
            sequence = " ".join(list(sequence))

        sequence_tokens = self.tokenize(sequence)[
            : min(self.max_length, kwargs.get("max_length", 512)) - 2
        ]

        if not is_bpe_tokenization(sequence_tokens):
            raise ValueError("The tokenizer seems not to be a BPE tokenizer.")
        tokenized_inputs = dict()
        tokenized_inputs["input_ids"] = self.base_tokenizer.convert_tokens_to_ids(
            sequence_tokens
        )
        tokenized_inputs["attention_mask"] = [1] * len(tokenized_inputs["input_ids"])

        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding="max_length",
            max_length=len(sequence_tokens),
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(config_or_model, **kwargs):
        """
        Create a BPE tokenizer from a pre-trained model.

        Args:
            config_or_model (str): Name or path of the pre-trained model
            **kwargs: Additional keyword arguments

        Returns:
            OmniBPETokenizer: Initialized BPE tokenizer

        Example:
            >>> tokenizer = OmniBPETokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            >>> print(type(tokenizer))
            <class 'omnigenome.src.tokenizer.bpe_tokenizer.OmniBPETokenizer'>
        """
        from transformers import AutoTokenizer

        self = OmniBPETokenizer(
            AutoTokenizer.from_pretrained(config_or_model, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        """
        Tokenize a sequence using the base BPE tokenizer.

        Args:
            sequence (str): Input sequence to tokenize
            **kwargs: Additional keyword arguments

        Returns:
            list: List of tokens

        Example:
            >>> sequence = "ACGUAGGUAUCGUAGA"
            >>> tokens = tokenizer.tokenize(sequence)
            >>> print(tokens[:5])
            ['▁A', 'C', 'G', 'U', 'A']
        """
        return self.base_tokenizer.tokenize(sequence)

    def encode(self, sequence, **kwargs):
        """
        Encode a sequence using the base BPE tokenizer.

        Args:
            sequence (str): Input sequence to encode
            **kwargs: Additional keyword arguments

        Returns:
            list: List of token IDs

        Raises:
            AssertionError: If the base tokenizer is not BPE-based

        Example:
            >>> sequence = "ACGUAGGUAUCGUAGA"
            >>> token_ids = tokenizer.encode(sequence)
            >>> print(len(token_ids))
            17
        """
        assert hasattr(
            self.base_tokenizer, "bpe"
        ), "The base tokenizer must be a BPE tokenizer."
        return self.base_tokenizer.encode(sequence, **kwargs)

    def decode(self, sequence, **kwargs):
        """
        Decode a sequence using the base BPE tokenizer.

        Args:
            sequence: Input sequence to decode (can be token IDs or tokens)
            **kwargs: Additional keyword arguments

        Returns:
            str: Decoded sequence

        Raises:
            AssertionError: If the base tokenizer is not BPE-based

        Example:
            >>> token_ids = [1, 2, 3, 4, 5]
            >>> sequence = tokenizer.decode(token_ids)
            >>> print(sequence)
            "ACGUAGGUAUCGUAGA"
        """
        assert hasattr(
            self.base_tokenizer, "bpe"
        ), "The base tokenizer must be a BPE tokenizer."
        return self.base_tokenizer.decode(sequence, **kwargs)
