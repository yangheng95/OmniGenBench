# -*- coding: utf-8 -*-
# file: omnigenome_wrapper.py
# time: 18:37 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import warnings

from transformers import AutoTokenizer

from ..misc.utils import env_meta_info, load_module_from_path


class OmniTokenizer:
    """
    A wrapper class for tokenizers to provide a consistent interface within OmniGenome.
    
    This class provides a unified interface for tokenizers in the OmniGenome framework.
    It wraps underlying tokenizers (typically from Hugging Face) and provides
    additional functionality for genomic sequence processing.
    
    The class handles various tokenization strategies and provides compatibility
    with different model architectures. It also supports custom tokenizer wrappers
    for specialized genomic tasks.
    
    Attributes:
        base_tokenizer: The underlying tokenizer instance (e.g., from Hugging Face).
        max_length (int): The default maximum sequence length.
        metadata (dict): Metadata about the tokenizer including version info.
        u2t (bool): Whether to convert 'U' to 'T'.
        t2u (bool): Whether to convert 'T' to 'U'.
        add_whitespace (bool): Whether to add whitespace between characters.
    """

    def __init__(self, base_tokenizer=None, max_length=512, **kwargs):
        """
        Initializes the tokenizer wrapper.

        Args:
            base_tokenizer: The underlying tokenizer instance (e.g., from Hugging Face).
            max_length (int): The default maximum sequence length. Defaults to 512.
            **kwargs: Additional keyword arguments.
                - u2t (bool): Whether to convert 'U' to 'T'. Defaults to False.
                - t2u (bool): Whether to convert 'T' to 'U'. Defaults to False.
                - add_whitespace (bool): Whether to add whitespace between characters.
                  Defaults to False.

        Example:
            >>> # Initialize with a Hugging Face tokenizer
            >>> from transformers import AutoTokenizer
            >>> base_tokenizer = AutoTokenizer.from_pretrained("model_name")
            >>> tokenizer = OmniTokenizer(base_tokenizer, max_length=512)
            
            >>> # Initialize with sequence conversion
            >>> tokenizer = OmniTokenizer(base_tokenizer, u2t=True)
        """
        self.metadata = env_meta_info()

        self.base_tokenizer = base_tokenizer
        self.max_length = max_length

        for key, value in kwargs.items():
            self.metadata[key] = value

        self.u2t = kwargs.get("u2t", False)
        self.t2u = kwargs.get("t2u", False)
        self.add_whitespace = kwargs.get("add_whitespace", False)

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        """
        Loads a tokenizer from a pre-trained model path.

        It attempts to load a custom tokenizer wrapper if `omnigenome_wrapper.py`
        is present in the model directory. Otherwise, it falls back to
        `transformers.AutoTokenizer`.

        Args:
            model_name_or_path (str): The name or path of the pre-trained model.
            **kwargs: Additional arguments for the tokenizer.

        Returns:
            OmniTokenizer: An instance of a tokenizer.

        Example:
            >>> # Load from a pre-trained model
            >>> tokenizer = OmniTokenizer.from_pretrained("model_name")
            
            >>> # Load with custom parameters
            >>> tokenizer = OmniTokenizer.from_pretrained("model_name", 
            ...                                          trust_remote_code=True)
        """
        wrapper_path = f"{model_name_or_path.rstrip('/')}/omnigenome_wrapper.py"
        try:
            tokenizer_cls = load_module_from_path(
                "OmniTokenizerWrapper", wrapper_path
            ).Tokenizer
            tokenizer = tokenizer_cls(
                AutoTokenizer.from_pretrained(model_name_or_path, **kwargs), **kwargs
            )
        except Exception as e:
            warnings.warn(
                f"No tokenizer wrapper found in {wrapper_path} -> Exception: {e}"
            )
            kwargs.pop("num_labels", None) # Remove num_labels if it exists, as it may not be applicable

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

        return tokenizer

    def save_pretrained(self, save_directory):
        """
        Saves the base tokenizer to a directory.

        Args:
            save_directory (str): The directory to save the tokenizer to.

        Example:
            >>> tokenizer.save_pretrained("./saved_tokenizer")
        """
        self.base_tokenizer.save_pretrained(save_directory)

    def __call__(self, *args, **kwargs):
        """
        Tokenizes inputs using the base tokenizer.

        This method provides a convenient interface for tokenization with
        sensible defaults for padding, truncation, and tensor conversion.

        Args:
            *args: Positional arguments for the base tokenizer.
            **kwargs: Keyword arguments for the base tokenizer.

        Returns:
            dict: The output from the base tokenizer, typically containing
                  'input_ids' and 'attention_mask'.

        Example:
            >>> # Tokenize a sequence
            >>> inputs = tokenizer("ATCGATCG")
            >>> print(inputs['input_ids'].shape)
        """
        padding = kwargs.pop("padding", True)
        truncation = kwargs.pop("truncation", True)
        max_length = kwargs.pop(
            "max_length", self.max_length if self.max_length else 512
        )
        return_tensor = kwargs.pop("return_tensors", "pt")
        return self.base_tokenizer(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensor,
            *args,
            **kwargs,
        )

    def tokenize(self, sequence, **kwargs):
        """
        Converts a sequence into a list of tokens. Must be implemented by subclasses.

        This method should be implemented by concrete tokenizer classes to define
        how sequences are tokenized for their specific use case.

        Args:
            sequence (str): The input sequence.
            **kwargs: Additional arguments.

        Returns:
            list: A list of tokens.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> # In a nucleotide tokenizer
            >>> tokens = tokenizer.tokenize("ATCGATCG")
            >>> print(tokens)  # ['A', 'T', 'C', 'G', 'A', 'T', 'C', 'G']
        """
        raise NotImplementedError(
            "The tokenize() function should be adapted for different models,"
            " please implement it for your model."
        )

    def encode(self, sequence, **kwargs):
        """
        Converts a sequence into a list of token IDs. Must be implemented by subclasses.

        This method should be implemented by concrete tokenizer classes to define
        how sequences are encoded into token IDs.

        Args:
            sequence (str): The input sequence.
            **kwargs: Additional arguments.

        Returns:
            list: A list of token IDs.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> # In a nucleotide tokenizer
            >>> token_ids = tokenizer.encode("ATCGATCG")
            >>> print(token_ids)  # [1, 2, 3, 4, 1, 2, 3, 4]
        """
        raise NotImplementedError(
            "The encode() function should be adapted for different models,"
            " please implement it for your model."
        )

    def decode(self, sequence, **kwargs):
        """
        Converts a list of token IDs back into a sequence. Must be implemented by subclasses.

        This method should be implemented by concrete tokenizer classes to define
        how token IDs are decoded back into sequences.

        Args:
            sequence (list): A list of token IDs.
            **kwargs: Additional arguments.

        Returns:
            str: The decoded sequence.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> # In a nucleotide tokenizer
            >>> sequence = tokenizer.decode([1, 2, 3, 4])
            >>> print(sequence)  # "ATCG"
        """
        raise NotImplementedError(
            "The decode() function should be adapted for different models,"
            " please implement it for your model."
        )

    def __getattribute__(self, item):
        """
        Custom attribute getter that falls back to the base tokenizer if an
        attribute is not found on the wrapper.

        This method provides transparent access to the base tokenizer's attributes,
        allowing the wrapper to be used as a drop-in replacement for the base tokenizer.

        Args:
            item (str): The attribute name to get.

        Returns:
            The attribute value from either the wrapper or the base tokenizer.

        Raises:
            AttributeError: If the attribute is not found on either the wrapper
                          or the base tokenizer.
        """
        try:
            return super().__getattribute__(item)
        except AttributeError:
            try:
                return self.base_tokenizer.__getattribute__(item)
            except (AttributeError, RecursionError) as e:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{item}'"
                ) from e
