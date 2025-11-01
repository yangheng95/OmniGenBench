# -*- coding: utf-8 -*-
# file: kmers_tokenizer.py
# time: 18:31 08/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import warnings

from ..abc.abstract_tokenizer import OmniTokenizer

warnings.filterwarnings("once")


class OmniKmersTokenizer(OmniTokenizer):
    """
    A k-mer based tokenizer for genomic sequences.

    This tokenizer breaks genomic sequences into overlapping k-mers and uses
    a base tokenizer to convert them into token IDs. It supports various
    k-mer sizes and overlap configurations for different genomic applications.

    Attributes:
        base_tokenizer: The underlying tokenizer for converting k-mers to IDs
        k: Size of k-mers
        overlap: Number of overlapping positions between consecutive k-mers
        max_length: Maximum sequence length for tokenization
        metadata: Dictionary containing tokenizer metadata

    Example:
        >>> from omnigenbench import OmniKmersTokenizer
        >>> from transformers import AutoTokenizer
        >>> base_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
        >>> tokenizer = OmniKmersTokenizer(base_tokenizer, k=4, overlap=2)
        >>> sequence = "ACGUAGGUAUCGUAGA"
        >>> tokens = tokenizer.tokenize(sequence)
        >>> print(tokens)
        [['ACGU', 'GUAG', 'UAGG', 'AGGU', 'GGUA', 'GUAU', 'UAUC', 'AUCG', 'UCGU', 'CGUA', 'GUAG', 'UAGA']]
    """

    def __init__(self, base_tokenizer=None, k=3, overlap=0, max_length=512, **kwargs):
        """
        Initialize the OmniKmersTokenizer.

        Args:
            base_tokenizer: The base tokenizer for converting k-mers to token IDs
            k (int, optional): Size of k-mers. Defaults to 3
            overlap (int, optional): Number of overlapping positions between consecutive k-mers. Defaults to 0
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512
            **kwargs: Additional keyword arguments passed to parent class
        """
        super(OmniKmersTokenizer, self).__init__(base_tokenizer, **kwargs)
        self.k = k
        self.overlap = overlap
        self.max_length = max_length
        self.metadata["tokenizer_name"] = self.__class__.__name__

    def __call__(self, sequence, **kwargs):
        """
        Tokenize a sequence or list of sequences into tokenized inputs.

        This method processes the input sequence(s) by first converting them to k-mers,
        then using the base tokenizer to convert k-mers to token IDs. It handles
        sequence preprocessing (U/T conversion) and adds special tokens.

        Args:
            sequence (str or list): Input sequence(s) to tokenize
            **kwargs: Additional keyword arguments including max_length

        Returns:
            dict: Dictionary containing tokenized inputs with keys 'input_ids' and 'attention_mask'

        Example:
            >>> sequence = "ACGUAGGUAUCGUAGA"
            >>> tokenized = tokenizer(sequence)
            >>> print(tokenized['input_ids'].shape)
            torch.Size([1, 14])
        """
        if self.u2t:
            sequence = "".join([seq.replace("U", "T").upper() for seq in sequence])
        if self.t2u:
            sequence = "".join([seq.replace("T", "U").upper() for seq in sequence])

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
                    f"Unknown tokens are more than 10% in the {i}th sequence, please check the tokenization process."
                )
        tokenized_inputs = self.base_tokenizer.pad(
            tokenized_inputs,
            padding="max_length",
            max_length=len(sequence_tokens),
            return_attention_mask=True,
            return_tensors="pt",
        )
        return tokenized_inputs

    @staticmethod
    def from_pretrained(config_or_model, **kwargs):
        """
        Create a k-mers tokenizer from a pre-trained model.

        Args:
            config_or_model (str): Name or path of the pre-trained model
            **kwargs: Additional keyword arguments

        Returns:
            OmniKmersTokenizer: Initialized k-mers tokenizer

        Example:
            >>> tokenizer = OmniKmersTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            >>> print(type(tokenizer))
            <class 'omnigenome.src.tokenizer.kmers_tokenizer.OmniKmersTokenizer'>
        """
        from transformers import AutoTokenizer

        self = OmniKmersTokenizer(
            AutoTokenizer.from_pretrained(config_or_model, **kwargs)
        )
        return self

    def tokenize(self, sequence, **kwargs):
        """
        Convert sequence(s) into k-mers.

        This method breaks the input sequence(s) into overlapping k-mers based on
        the configured k-mer size and overlap parameters.

        Args:
            sequence (str or list): Input sequence(s) to convert to k-mers
            **kwargs: Additional keyword arguments

        Returns:
            list: List of k-mer lists for each input sequence

        Example:
            >>> sequence = "ACGUAGGUAUCGUAGA"
            >>> k_mers = tokenizer.tokenize(sequence)
            >>> print(k_mers[0][:3])
            ['ACGU', 'GUAG', 'UAGG']
        """
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
        """
        Encode input IDs using the base tokenizer.

        Args:
            input_ids: Input IDs to encode
            **kwargs: Additional keyword arguments

        Returns:
            Encoded input IDs
        """
        return self.base_tokenizer.encode(input_ids, **kwargs)

    def decode(self, input_ids, **kwargs):
        """
        Decode input IDs using the base tokenizer.

        Args:
            input_ids: Input IDs to decode
            **kwargs: Additional keyword arguments

        Returns:
            Decoded sequence
        """
        return self.base_tokenizer.decode(input_ids, **kwargs)

    def encode_plus(self, sequence, **kwargs):
        """
        Encode a sequence with additional information.

        This method is not yet implemented for k-mers tokenizer.

        Args:
            sequence: Input sequence
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: This method is not implemented yet
        """
        raise NotImplementedError("The encode_plus() function is not implemented yet.")


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # RNA = "ACGUAGGUAUCGUAGA"
    # # base_tokenizer_name = 'bert-base-cased'
    # base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    # base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    # tokenizer = KmersTokenizer(base_tokenizer)
    # tokens = tokenizer.tokenize(RNA)
    # fprint(tokens)
    # tokenized_inputs = tokenizer(RNA)
    # fprint(tokenized_inputs)

    RNA = "ACGUAGGUAUCGUAGA"
    # base_tokenizer_name = 'bert-base-cased'
    base_tokenizer_name = "facebook/esm2_t12_35M_UR50D"
    base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    tokenizer = OmniKmersTokenizer(base_tokenizer, k=4, overlap=2, max_length=512)
    tokens = tokenizer.tokenize(RNA)
    print(tokens)
    tokenized_inputs = tokenizer(RNA)
    print(tokenized_inputs)
