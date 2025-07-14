# -*- coding: utf-8 -*-
# file: lora_model.py
# time: 12:36 11/06/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
Low-Rank Adaptation (LoRA) models for OmniGenome.

This module provides LoRA implementation for efficient fine-tuning of large
genomic language models. LoRA reduces the number of trainable parameters
by adding low-rank adaptation layers to existing model weights.
"""
from torch import nn
from ...src.misc.utils import fprint


def find_linear_target_modules(model, keyword_filter=None, use_full_path=True):
    """
    Find linear modules in a model that can be targeted for LoRA adaptation.

    This function searches through a model's modules to identify linear layers
    that can be adapted using LoRA. It supports filtering by keyword patterns
    to target specific types of layers.

    Args:
        model: The model to search for linear modules
        keyword_filter (str, list, tuple, optional): Keywords to filter modules by name
        use_full_path (bool): Whether to return full module paths or just names (default: True)

    Returns:
        list: Sorted list of linear module names that can be targeted for LoRA

    Raises:
        TypeError: If keyword_filter is not None, str, or a list/tuple of str
    """
    import re
    from torch import nn

    if keyword_filter is not None:
        if isinstance(keyword_filter, str):
            keyword_filter = [keyword_filter]
        elif not isinstance(keyword_filter, (list, tuple)):
            raise TypeError("keyword_filter must be None, str, or a list/tuple of str")

        pattern = "|".join(map(re.escape, keyword_filter))

    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if keyword_filter is None or re.search(pattern, name, re.IGNORECASE):
                linear_modules.add(name if use_full_path else name.split(".")[-1])

    return sorted(linear_modules)


def auto_lora_model(model, **kwargs):
    """
    Automatically create a LoRA-adapted model.

    This function automatically identifies suitable target modules and creates
    a LoRA-adapted version of the input model. It handles configuration
    setup and parameter freezing for efficient fine-tuning.

    Args:
        model: The base model to adapt with LoRA
        **kwargs: Additional LoRA configuration parameters

    Returns:
        The LoRA-adapted model

    Raises:
        AssertionError: If no target modules are found for LoRA injection
    """
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig

    # A bad case for the EVO-1 model, which has a custom config class
    ######################
    if hasattr(model, "config") and not isinstance(model.config, PretrainedConfig):
        delattr(model.config, "Loader")
        model.config = PretrainedConfig.from_dict(dict(model.config))
    #######################

    target_modules = kwargs.pop("target_modules", None)
    use_rslora = kwargs.pop("use_rslora", True)
    bias = kwargs.pop("bias", "none")
    r = kwargs.pop("r", 32)
    lora_alpha = kwargs.pop("lora_alpha", 256)
    lora_dropout = kwargs.pop("lora_dropout", 0.1)

    if target_modules is None:
        target_modules = find_linear_target_modules(
            model, keyword_filter=kwargs.get("keyword_filter", None)
        )
    assert target_modules is not None, "No target modules found for LoRA injection."
    config = LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_rslora=use_rslora,
        **kwargs,
    )

    for param in model.parameters():
        param.requires_grad = False

    lora_model = get_peft_model(model, config)
    trainable_params, all_param = lora_model.get_nb_trainable_parameters()
    fprint(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
        f" || trainable%: {100 * trainable_params / all_param:.4f}"
    )
    return lora_model


class OmniLoraModel(nn.Module):
    """
    LoRA-adapted model for OmniGenome.

    This class provides a wrapper around LoRA-adapted models, enabling
    efficient fine-tuning of large genomic language models while maintaining
    compatibility with the OmniGenome framework.

    Attributes:
        lora_model: The underlying LoRA-adapted model
        config: Model configuration
        device: Device the model is running on
        dtype: Data type of the model parameters
    """

    def __init__(self, model, **kwargs):
        """
        Initialize the LoRA-adapted model.

        Args:
            model: The base model to adapt with LoRA
            **kwargs: LoRA configuration parameters

        Raises:
            ValueError: If no target modules are specified for LoRA injection
        """
        super(OmniLoraModel, self).__init__()
        target_modules = kwargs.get("target_modules", None)
        if target_modules is None:
            raise ValueError(
                "No target modules found for LoRA injection. To perform LoRA adaptation fine-tuning, "
                "please specify the target modules using the 'target_modules' argument. "
                "The target modules depend on the model architecture, such as 'query', 'value', etc. "
            )

        self.lora_model = auto_lora_model(model, **kwargs)

        fprint(
            "To reduce GPU memory occupation, "
            "you should avoid include non-trainable parameters into optimizers, "
            "e.g.,  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...), "
            "AVOIDING: optimizer = torch.optim.AdamW(model.parameters(), ...)"
        )

        self.config = model.config
        self.to("cpu")  # Move the model to CPU initially
        fprint(
            "LoRA model initialized with the following configuration:\n",
            self.lora_model,
        )

    def to(self, *args, **kwargs):
        """
        Move the model to a specific device and data type.

        This method overrides the default to() method to ensure the LoRA model
        and its components are properly moved to the target device and dtype.

        Args:
            *args: Device specification (e.g., 'cuda', 'cpu')
            **kwargs: Additional arguments including dtype

        Returns:
            self: The model instance
        """
        self.lora_model.to(*args, **kwargs)
        try:
            # For evo-1 and similar models, we need to set the device and dtype
            for param in self.parameters():
                self.device = param.device
                self.dtype = param.dtype
                break
            for module in self.lora_model.modules():
                module.device = self.device
                if hasattr(module, "dtype"):
                    module.dtype = self.dtype
        except Exception as e:
            pass  # Ignore errors if parameters are not available
        return self

    def forward(self, *args, **kwargs):
        """
        Forward pass through the LoRA model.

        Args:
            *args: Positional arguments for the forward pass
            **kwargs: Keyword arguments for the forward pass

        Returns:
            The output from the LoRA model
        """
        return self.lora_model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Generate predictions using the LoRA model.

        Args:
            *args: Positional arguments for prediction
            **kwargs: Keyword arguments for prediction

        Returns:
            Model predictions
        """
        return self.lora_model.base_model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        Save the LoRA model.

        Args:
            *args: Positional arguments for saving
            **kwargs: Keyword arguments for saving

        Returns:
            Result of the save operation
        """
        return self.lora_model.base_model.save(*args, **kwargs)

    def model_info(self):
        """
        Get information about the LoRA model.

        Returns:
            Model information from the base model
        """
        return self.lora_model.base_model.model_info()

    def set_loss_fn(self, fn):
        """
        Set the loss function for the LoRA model.

        Args:
            fn: Loss function to set

        Returns:
            Result of setting the loss function
        """
        return self.lora_model.base_model.set_loss_fn(fn)

    def last_hidden_state_forward(self, **kwargs):
        """
        Forward pass to get the last hidden state.

        Args:
            **kwargs: Keyword arguments for the forward pass

        Returns:
            Last hidden state from the base model
        """
        return self.lora_model.base_model.last_hidden_state_forward(**kwargs)

    def tokenizer(self):
        """
        Get the tokenizer from the base model.

        Returns:
            The tokenizer from the base model
        """
        return self.lora_model.base_model.tokenizer

    def config(self):
        """
        Get the configuration from the base model.

        Returns:
            The configuration from the base model
        """
        return self.lora_model.base_model.config

    def model(self):
        """
        Get the base model.

        Returns:
            The base model
        """
        return self.lora_model.base_model.model
