# -*- coding: utf-8 -*-
# file: abstract_tokenizer.py
# time: 18:37 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import os
import warnings

from transformers import AutoTokenizer

from ..misc.utils import env_meta_info, load_module_from_path


class OmniTokenizer:
    """
    Abstract base class providing a unified interface for tokenizers in the OmniGenBench
    framework. This class wraps underlying tokenizers (typically from HuggingFace) and
    provides genomic-specific preprocessing functionality for biological sequence analysis.

    **Design Pattern**: This class implements the Wrapper pattern (also known as Adapter),
    providing a consistent API while delegating core tokenization to specialized implementations.
    It adds genomic-specific preprocessing (RNA-to-DNA conversion, whitespace insertion,
    sequence normalization) while maintaining compatibility with HuggingFace's tokenizer ecosystem.

    **Architecture**: The tokenizer stack consists of three layers:

    1. **Base Tokenizer**: HuggingFace AutoTokenizer or custom implementation providing
       vocabulary, encoding/decoding primitives, and special token handling.

    2. **OmniTokenizer Wrapper** (this class): Adds genomic preprocessing, metadata tracking,
       and unified API across different tokenizer types.

    3. **Custom Wrappers**: Optional task-specific wrappers (loaded from omnigenome_wrapper.py
       if present in model directory) for specialized preprocessing logic.

    **Genomic-Specific Features**:

    - **Sequence Normalization**: Automatic uppercase conversion and nucleotide standardization
      (converts lowercase to uppercase, handles ambiguous nucleotide codes).

    - **RNA/DNA Conversion**: Bidirectional U↔T conversion via u2t and t2u flags. Essential
      when applying models trained on one sequence type to another (e.g., using DNA models
      for RNA data).

    - **Whitespace Injection**: Optional character separation for character-level models.
      Converts "ATCG" → "A T C G" for models trained with spaced sequences.

    - **Special Token Handling**: Automatic insertion of [CLS], [SEP], [PAD], [MASK] tokens
      according to model requirements. Handles both BERT-style (e.g., [CLS] seq [SEP]) and
      GPT-style (seq [EOS]) conventions.

    - **K-mer Tokenization**: Support for overlapping k-mer segmentation (k=3,4,5,6) for
      capturing local sequence patterns. Example: "ATCGATCG" with k=3 → "ATC TCG CGA GAT ATC TCG".

    - **Codon-Aware Tokenization**: Specialized handling for protein-coding sequences with
      triplet nucleotide units, preserving reading frame information.

    - **Structure-Informed Tokenization**: Optional integration with RNA secondary structure
      for structure-aware models (dot-bracket notation encoding).

    **Integration with Model Loading**: When loading models via ModelHub or from_pretrained(),
    the framework first checks for custom tokenizer wrappers (omnigenome_wrapper.py) in the
    model directory, falling back to standard AutoTokenizer if not found. This enables
    model-specific preprocessing without modifying core code.

    **Common Tokenizer Types**:

    - **OmniSingleNucleotideTokenizer**: Character-level tokenization (vocab size ~10)
    - **OmniKmersTokenizer**: K-mer based tokenization (vocab size 4^k, typically 64-4096)
    - **OmniBPETokenizer**: Byte-Pair Encoding for learned subword units (vocab size 1000-50000)

    Attributes:
        base_tokenizer: Underlying tokenizer instance (e.g., from HuggingFace Transformers).
            Provides vocabulary, encoding primitives, and special token definitions.
            Can be any object implementing encode(), decode(), and __call__() methods.

        max_length (int): Default maximum sequence length for tokenization. Can be overridden
            in individual tokenization calls. Sequences longer than this are truncated.
            Typical values: 512 (short sequences), 2048 (medium), 10000+ (long genomic regions).

        metadata (dict): Framework metadata including version information and custom attributes.
            Automatically populated with tokenizer type, version, timestamp, etc.

        u2t (bool): Whether to convert 'U' (uracil) to 'T' (thymine) for RNA→DNA conversion.
            Useful when training DNA models on RNA data or applying DNA-trained models to
            RNA sequences. Default False.

        t2u (bool): Whether to convert 'T' to 'U' for DNA→RNA conversion. Useful for RNA
            structure prediction models trained on DNA sequences, or when applying RNA models
            to DNA data. Default False.

        add_whitespace (bool): Whether to insert spaces between characters for character-level
            tokenization. Required for some BERT-style models trained on spaced sequences.
            Example: "ATCG" becomes "A T C G". Default False.

        trust_remote_code (bool): Whether to trust remote code when loading tokenizers from
            HuggingFace Hub. Default True. Set to False in security-critical environments.

    Note:
        - Set u2t=True when using DNA models on RNA sequences
        - Set t2u=True when using RNA models on DNA sequences
        - Never set both u2t=True and t2u=True simultaneously (results undefined)
        - add_whitespace should match the training configuration of the model
    """

    def __init__(self, base_tokenizer=None, max_length=512, **kwargs):
        """
        Initializes the tokenizer wrapper with genomic-specific preprocessing options.

        Args:
            base_tokenizer: Underlying tokenizer instance providing vocabulary and encoding.
                Typically a HuggingFace AutoTokenizer, but can be any object implementing
                encode(), decode(), and __call__() methods. If None, a minimal tokenizer
                will be created (for custom implementations).
            max_length (int): Default maximum sequence length for tokenization. Individual
                tokenization calls can override this value. Sequences longer than this
                length are truncated. Defaults to 512.
            **kwargs: Additional keyword arguments for genomic preprocessing:
                - u2t (bool): Convert RNA sequences to DNA (U→T). Useful when applying
                  DNA-trained models to RNA sequences. Defaults to False.
                - t2u (bool): Convert DNA sequences to RNA (T→U). Useful when applying
                  RNA-trained models to DNA sequences. Defaults to False.
                - add_whitespace (bool): Insert spaces between characters for character-level
                  models trained on spaced sequences (e.g., "ATCG" → "A T C G"). Defaults
                  to False. Required for some BERT variants.
                - trust_remote_code (bool): Whether to trust remote code when loading from
                  HuggingFace Hub. Defaults to True. Set to False in security-sensitive
                  environments.

                Additional custom attributes are stored in metadata for access by
                downstream components.

        Example:
            >>> # Pattern 1: Initialize with a HuggingFace tokenizer
            >>> from transformers import AutoTokenizer
            >>> base_tokenizer = AutoTokenizer.from_pretrained("yangheng/OmniGenome-186M")
            >>> tokenizer = OmniTokenizer(base_tokenizer, max_length=512)

            >>> # Pattern 2: Initialize with RNA-to-DNA conversion
            >>> tokenizer = OmniTokenizer(base_tokenizer, u2t=True)
            >>> # RNA sequences like "AUGC" will be converted to "ATGC" before tokenization

            >>> # Pattern 3: Initialize with whitespace insertion for character-level models
            >>> tokenizer = OmniTokenizer(base_tokenizer, add_whitespace=True)
            >>> # "ATCG" → "A T C G" before tokenization

            >>> # Pattern 4: Initialize with custom max_length
            >>> tokenizer = OmniTokenizer(base_tokenizer, max_length=1024)
            >>> # Suitable for long genomic sequences (e.g., whole genes)
        """
        self.metadata = env_meta_info()

        self.base_tokenizer = base_tokenizer
        if self.base_tokenizer is None:
            warnings.warn(
                "Base tokenizer is None. Please ensure to implement tokenizer with encode(), decode() and __call__() methods in subclasses."
            )
        self.max_length = max_length

        for key, value in kwargs.items():
            self.metadata[key] = value

        self.u2t = kwargs.get("u2t", False)
        self.t2u = kwargs.get("t2u", False)
        self.add_whitespace = kwargs.get("add_whitespace", False)

    @staticmethod
    def from_pretrained(config_or_model, **kwargs):
        """
        Loads a tokenizer from a pre-trained model path.

        Args:
            config_or_model (str): The name or path of the pre-trained model.
            **kwargs: Additional arguments for the tokenizer.

        Returns:
            OmniTokenizer: An instance of a tokenizer.

        Example:
            >>> # Load from a pre-trained model
            >>> tokenizer = OmniTokenizer.from_pretrained("model_name")
            >>> # Load with custom parameters
            >>> tokenizer = OmniTokenizer.from_pretrained("model_name", trust_remote_code=True)
        """
        kwargs.pop("num_labels", None)  # Seems we don't need num_labels here

        wrapper_path = f"{config_or_model.rstrip('/')}/omnigenome_wrapper.py"
        if os.path.exists(wrapper_path):
            tokenizer_cls = load_module_from_path(
                "OmniTokenizerWrapper", wrapper_path
            ).Tokenizer
            tokenizer = tokenizer_cls(
                AutoTokenizer.from_pretrained(config_or_model, **kwargs), **kwargs
            )
        else:
            if "multimolecule" in config_or_model:
                from multimolecule import RnaTokenizer

                tokenizer = RnaTokenizer.from_pretrained(config_or_model, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(config_or_model, **kwargs)

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
