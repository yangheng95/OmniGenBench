# -*- coding: utf-8 -*-
# file: lora_model.py
# time: 12:36 11/06/2025
# author: YENG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
This module provides Low-Rank Adaptation (LoRA) implementation for efficient fine-tuning of large
genomic language models. LoRA reduces the number of trainable parameters
by adding low-rank adaptation layers to existing model weights.
"""
from torch import nn
from ...src.misc.utils import fprint


def find_linear_target_modules(model, keyword_filter=None, use_full_path=True):
    """
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

    pattern = None
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
    import torch

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
        fprint(
            "No target modules found for LoRA injection. To perform LoRA adaptation fine-tuning, "
            "please specify the target modules using the 'target_modules' argument. "
            "The target modules depend on the model architecture, such as 'query', 'value', etc. ",
            "If you are unsure about the target modules, we use 'find_linear_target_modules' function ",
            "to automatically identify suitable modules based on keywords.",
        )
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

    # More aggressive parameter freezing to save memory
    # First completely detach all parameters from computation graph
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.requires_grad = False
            p.requires_grad_(False)
            # Ensure parameter is detached from computation graph
            p.grad = None

    # Build Peft LoRA model
    lora_model = get_peft_model(model, config)

    # Freeze base model parameters more aggressively
    with torch.no_grad():
        for name, p in lora_model.named_parameters():
            if "lora_" not in name and "modules_to_save" not in name:
                # This is a base model parameter, freeze it completely
                p.requires_grad = False
                p.requires_grad_(False)
                p.grad = None
            elif not p.requires_grad:
                # Ensure frozen parameters stay frozen
                p.requires_grad_(False)
                p.grad = None

    # Enable gradient checkpointing for the base model to save activation memory
    if hasattr(lora_model, "base_model") and hasattr(
        lora_model.base_model, "gradient_checkpointing_enable"
    ):
        try:
            lora_model.base_model.gradient_checkpointing_enable()
            fprint(
                "Gradient checkpointing enabled for base model to save activation memory"
            )
        except Exception as e:
            fprint(f"Could not enable gradient checkpointing: {e}")

    trainable_params, all_param = lora_model.get_nb_trainable_parameters()
    fprint(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d}"
        f" || trainable%: {100 * trainable_params / all_param:.4f}"
    )

    # Clear any cached gradients
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return lora_model


class OmniLoraModel(nn.Module):
    """
    Wrapper around a LoRA-adapted model.
    """

    def __init__(self, model, **kwargs):
        """
        Initialize the LoRA-adapted model.
        """
        import torch

        super(OmniLoraModel, self).__init__()

        lora_config = kwargs.pop("lora_config", {})
        self.lora_model = auto_lora_model(model, **lora_config)

        # Store original model reference for efficient memory management
        self._original_model = model

        # Track whether we've logged memory-related flags already
        self._flags_logged = False

        # Cache original transformer flags to toggle for training/eval
        self._orig_use_cache = None
        self._orig_output_hidden_states = None
        self._orig_output_attentions = None

        fprint(
            "To reduce GPU memory occupation, "
            + "avoid including non-trainable parameters in optimizers, e.g., "
            + "optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...), "
            + "AVOID: optimizer = torch.optim.AdamW(model.parameters(), ...)"
        )

        self.config = model.config

        # Optional: enable gradient checkpointing for activation memory savings
        try:
            if hasattr(self.lora_model, "gradient_checkpointing_enable"):
                self.lora_model.gradient_checkpointing_enable()
                fprint("Gradient checkpointing enabled")
            elif hasattr(self.lora_model, "base_model") and hasattr(
                self.lora_model.base_model, "gradient_checkpointing_enable"
            ):
                self.lora_model.base_model.gradient_checkpointing_enable()
                fprint("Gradient checkpointing enabled (base_model)")
        except Exception:
            pass

        # Proactively clear any transient CUDA caches
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        fprint("LoRA model initialized:\n", self.lora_model)

    def to(self, *args, **kwargs):
        """
        Move the LoRA model to a specified device and/or data type with intelligent state management.

        This method extends PyTorch's standard .to() functionality by ensuring that both the
        underlying LoRA model and wrapper state (device/dtype tracking) remain consistent
        after device or precision changes. It safely handles device transfers while
        maintaining model integrity.

        Args:
            *args: Positional arguments passed to the underlying model's .to() method.
                Common usage:
                - device (str/torch.device): Target device ('cuda', 'cpu', 'cuda:0', etc.)
                - dtype (torch.dtype): Target data type (torch.float16, torch.float32, etc.)
            **kwargs: Keyword arguments passed to the underlying model's .to() method.
                Supports all standard PyTorch .to() parameters.

        Returns:
            OmniLoraModel: Returns self for method chaining, following PyTorch convention.

        Side Effects:
            - Updates internal device and dtype tracking attributes
            - Moves all model parameters and buffers to the specified device/dtype
            - Synchronizes device information across all model modules

        Examples:
            Move to GPU:
            >>> model = model.to('cuda')
            >>> model = model.to(torch.device('cuda:0'))

            Change precision:
            >>> model = model.to(torch.float16)  # Convert to half precision
            >>> model = model.to(dtype=torch.bfloat16)  # Convert to bfloat16

            Combined device and dtype:
            >>> model = model.to('cuda', dtype=torch.float16)

            Method chaining:
            >>> model = OmniLoraModel(base_model).to('cuda').train()

        Note:
            - Device/dtype information is automatically tracked for internal consistency
            - Exception handling ensures robustness if parameter introspection fails
            - All modules receive updated device information for framework compatibility
            - LoRA adapters maintain the same precision as the base model after transfer
        """
        self.lora_model.to(*args, **kwargs)
        try:
            for param in self.parameters():
                self.device = param.device
                self.dtype = param.dtype
                break
            for module in self.lora_model.modules():
                module.device = self.device
                if hasattr(module, "dtype"):
                    module.dtype = self.dtype
        except Exception:
            pass
        return self

    def forward(self, *args, **kwargs):
        """
        Perform a forward pass through the LoRA-adapted model.

        This method delegates the forward computation to the underlying LoRA model,
        which automatically combines the frozen base model outputs with the LoRA
        adaptation outputs. The forward pass is mathematically equivalent to:
        output = BaseModel(x) + LoRA_adaptation(x)

        Args:
            *args: Positional arguments passed to the underlying model's forward method.
                Typically includes input tensors (input_ids, attention_mask, etc.).
            **kwargs: Keyword arguments passed to the underlying model's forward method.
                Model-specific parameters like labels, output_hidden_states, etc.

        Returns:
            Model outputs in the same format as the base model, but incorporating
            LoRA adaptations. The exact return type depends on the base model
            architecture (e.g., BaseModelOutput, SequenceClassifierOutput).

        Examples:
            Basic forward pass:
            >>> outputs = model(input_ids, attention_mask=attention_mask)

            With additional parameters:
            >>> outputs = model(
            ...     input_ids=input_ids,
            ...     attention_mask=attention_mask,
            ...     labels=labels,
            ...     output_hidden_states=True
            ... )

        Note:
            - LoRA adaptations are automatically applied during the forward pass
            - No manual intervention needed to combine base and adaptation outputs
            - Maintains full compatibility with the original model's forward signature
        """
        return self.lora_model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Generate predictions using the LoRA-adapted model through the base model interface.

        This method provides access to the base model's prediction functionality while
        incorporating LoRA adaptations. It's particularly useful for inference tasks
        where the base model has specialized prediction methods.

        Args:
            *args: Positional arguments passed to the base model's predict method.
            **kwargs: Keyword arguments passed to the base model's predict method.

        Returns:
            Predictions from the base model, enhanced with LoRA adaptations.
            The format depends on the specific base model's predict implementation.

        Examples:
            Generate predictions:
            >>> predictions = model.predict(input_sequences)

            With custom parameters:
            >>> predictions = model.predict(
            ...     sequences,
            ...     max_length=512,
            ...     temperature=0.7
            ... )

        Note:
            - Delegates to the base model's predict method while maintaining LoRA adaptations
            - Useful for models with specialized inference interfaces
            - LoRA adaptations are automatically included in the prediction process
        """
        return self.lora_model.base_model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        Save the LoRA-adapted model using the base model's save functionality.

        This method delegates saving operations to the base model while preserving
        LoRA adapter information. The exact saving behavior depends on the base
        model's implementation.

        Args:
            *args: Positional arguments passed to the base model's save method.
            **kwargs: Keyword arguments passed to the base model's save method.

        Returns:
            Result of the base model's save operation.

        Examples:
            Save model:
            >>> model.save('path/to/save/directory')

            Save with custom parameters:
            >>> model.save(
            ...     save_directory='./checkpoints',
            ...     save_config=True,
            ...     save_tokenizer=True
            ... )

        Note:
            - For saving LoRA adapters specifically, use PEFT's save_pretrained() method
            - This method saves the complete model state including LoRA adaptations
            - Check base model documentation for specific save parameters
        """
        return self.lora_model.base_model.save(*args, **kwargs)

    def model_info(self):
        """
        Get detailed information about the underlying base model.

        Returns comprehensive information about the model architecture, configuration,
        and other metadata through the base model's info interface.

        Returns:
            Model information from the base model, typically including architecture
            details, parameter counts, configuration settings, etc.

        Examples:
            Get model information:
            >>> info = model.model_info()
            >>> print(info)  # Display model architecture and stats

        Note:
            - Information reflects the base model architecture
            - LoRA-specific details may not be included (use print(model) for LoRA info)
            - Useful for understanding the underlying model structure
        """
        return self.lora_model.base_model.model_info()

    def set_loss_fn(self, fn):
        """
        Set a custom loss function for the base model.

        This method allows configuration of specialized loss functions through
        the base model's interface, useful for custom training objectives.

        Args:
            fn (callable): Loss function to be used by the base model.
                Should follow PyTorch loss function conventions.

        Returns:
            Result of setting the loss function on the base model.

        Examples:
            Set custom loss function:
            >>> custom_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            >>> model.set_loss_fn(custom_loss)

        Note:
            - Loss function applies to the combined base + LoRA model outputs
            - Delegates to the base model's loss function setting mechanism
            - LoRA adaptations are included in loss computation automatically
        """
        return self.lora_model.base_model.set_loss_fn(fn)

    def last_hidden_state_forward(self, **kwargs):
        """
        Perform forward pass and return the last hidden state from the base model.

        This method provides access to intermediate representations from the base
        model while incorporating LoRA adaptations, useful for feature extraction
        and analysis tasks.

        Args:
            **kwargs: Keyword arguments passed to the base model's last_hidden_state_forward method.

        Returns:
            Last hidden state tensor with LoRA adaptations applied.
            Shape typically [batch_size, sequence_length, hidden_size].

        Examples:
            Get hidden states:
            >>> hidden_states = model.last_hidden_state_forward(
            ...     input_ids=input_ids,
            ...     attention_mask=attention_mask
            ... )

        Note:
            - Hidden states include the effects of LoRA adaptations
            - Useful for feature extraction and representation analysis
            - Maintains compatibility with base model's hidden state interface
        """
