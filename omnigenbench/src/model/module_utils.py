# -*- coding: utf-8 -*-
# file: module_utils.py
# time: 22:53 18/07/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
This module provides utility classes and functions for handling model inputs,
pooling operations, and attention mechanisms used across different OmniGenome model types.
"""
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertPooler
from transformers.tokenization_utils_base import BatchEncoding


class OmniPooling(torch.nn.Module):
    """
    A flexible pooling layer for OmniGenome models that handles different input formats.

    This class provides a unified interface for pooling operations across different
    model architectures, supporting both causal language models and encoder-based models.
    It can handle various input formats including tuples, dictionaries, BatchEncoding
    objects, and tensors.

    Attributes:
        config: Model configuration object containing architecture and tokenizer settings
        pooler: BertPooler instance for non-causal models, None for causal models
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initialize the OmniPooling layer.

        Args:
            config: Model configuration object containing architecture information
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.pooler = BertPooler(self.config) if not self._is_causal_lm() else None

    def forward(self, inputs, last_hidden_state):
        """
        Perform pooling operation on the last hidden state.

        This method handles different input formats and applies appropriate pooling:
        - For causal language models: Uses the last non-padded token
        - For encoder models: Uses the BertPooler

        Args:
            inputs: Input data in various formats (tuple, dict, BatchEncoding, or tensor)
            last_hidden_state (torch.Tensor): Hidden states from the model [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Pooled representation [batch_size, hidden_size]

        Raises:
            ValueError: If input format is not supported or cannot be parsed
        """
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
        elif isinstance(inputs, BatchEncoding) or isinstance(inputs, dict):
            input_ids = inputs["input_ids"]
            attention_mask = (
                inputs["attention_mask"] if "attention_mask" in inputs else None
            )
        elif isinstance(inputs, torch.Tensor):
            shape = inputs.shape
            try:
                if len(shape) == 3:
                    # compatible with hf_trainer in AutoBenchmark
                    if shape[1] == 2:
                        input_ids = inputs[:, 0]
                        attention_mask = inputs[:, 1]
                    else:
                        input_ids = inputs[0]
                        attention_mask = inputs[1] if len(inputs) > 1 else None
                elif len(shape) == 2:
                    input_ids = inputs
                    attention_mask = None
            except:
                raise ValueError(
                    f"Failed to get the input_ids and attention_mask from the inputs, got shape {shape}."
                )
        else:
            raise ValueError(
                f"The inputs should be a tuple, BatchEncoding or a dictionary-like object, got {type(inputs)}."
            )

        if not self.pooler:
            pad_token_id = getattr(self.config, "pad_token_id", -100)
            sequence_lengths = input_ids.ne(pad_token_id).sum(dim=1) - 1
            last_hidden_state = last_hidden_state[
                torch.arange(input_ids.size(0), device=last_hidden_state.device),
                sequence_lengths,
            ]
        else:
            last_hidden_state = self.pooler(last_hidden_state)

        return last_hidden_state

    def _is_causal_lm(self):
        """
        Check if the model is a causal language model.

        Determines if the model architecture is causal based on the configuration.

        Returns:
            bool: True if the model is a causal language model, False otherwise
        """
        if (
            hasattr(self.config, "architectures")
            and "CausalLM" in str(self.config.architectures)
        ) or (
            hasattr(self.config, "auto_map") and "CausalLM" in str(self.config.auto_map)
        ):
            return True
        else:
            return False


# class InteractingAttention(nn.Module):
#     def __init__(self, embed_size, num_heads=12):
#         super(InteractingAttention, self).__init__()
#         self.num_heads = num_heads
#         self.embed_size = embed_size
#
#         assert embed_size % num_heads == 0, "Embedding size should be divisible by number of heads"
#
#         self.head_dim = embed_size // num_heads
#
#         self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
#         self.layer_norm = nn.LayerNorm(num_heads * self.head_dim, eps=1e-6)
#
#         self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
#
#     # def forward(self, query, keys, values):
#     def forward(self, query, keys, values):
#
#         N = query.shape[0]
#         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
#
#         # Split embedding into self.num_heads pieces
#         values = values.reshape(N, value_len, self.num_heads, self.head_dim)
#         keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
#         queries = query.reshape(N, query_len, self.num_heads, self.head_dim)
#
#         values = self.values(values)  # (N, value_len, heads, head_dim)
#         keys = self.keys(keys)  # (N, key_len, heads, head_dim)
#         queries = self.queries(queries)  # (N, query_len, heads, head_dim)
#
#         energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
#
#         attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)
#
#         out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
#             N, query_len, self.num_heads * self.head_dim
#         )
#         out = self.layer_norm(out + query)
#         out = self.fc_out(out)
#         out = self.layer_norm(out + query)
#         return out


class InteractingAttention(nn.Module):
    """
    An interacting attention mechanism for sequence modeling.

    This class implements a multi-head attention mechanism with residual connections
    and layer normalization. It's designed for processing sequences where different
    parts of the sequence need to interact with each other.

    Attributes:
        attention: Multi-head attention layer
        layer_norm: Layer normalization for residual connections
        fc_out: Output projection layer
    """

    def __init__(self, embed_size, num_heads=24):
        """
        Initialize the InteractingAttention module.

        Args:
            embed_size (int): Size of the embedding dimension
            num_heads (int): Number of attention heads (default: 24)

        Raises:
            AssertionError: If embed_size is not divisible by num_heads
        """
        super(InteractingAttention, self).__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size should be divisible by number of heads"

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, keys, values):
        """
        Forward pass through the interacting attention mechanism.

        Args:
            query (torch.Tensor): Query tensor [batch_size, query_len, embed_size]
            keys (torch.Tensor): Key tensor [batch_size, key_len, embed_size]
            values (torch.Tensor): Value tensor [batch_size, value_len, embed_size]

        Returns:
            torch.Tensor: Output tensor with same shape as query
        """
        att_output, _ = self.attention(query, keys, values)

        query = self.layer_norm(att_output + query)

        output = self.fc_out(query)
        output = self.layer_norm(output + query)

        return output
