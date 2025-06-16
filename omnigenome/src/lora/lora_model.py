# -*- coding: utf-8 -*-
# file: lora_model.py
# time: 12:36 11/06/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import torch
from torch import nn
from omnigenome.src.misc.utils import fprint

def find_linear_target_modules(model, keyword_filter=None, use_full_path=True):
    import re
    from torch import nn

    if keyword_filter is not None:
        if isinstance(keyword_filter, str):
            keyword_filter = [keyword_filter]
        elif not isinstance(keyword_filter, (list, tuple)):
            raise TypeError("keyword_filter must be None, str, or a list/tuple of str")

        pattern = '|'.join(map(re.escape, keyword_filter))

    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if keyword_filter is None or re.search(pattern, name, re.IGNORECASE):
                linear_modules.add(name if use_full_path else name.split('.')[-1])

    return sorted(linear_modules)

def auto_lora_model(model, **kwargs):
    from peft import LoraConfig, get_peft_model
    from transformers import PretrainedConfig

    # A bad case for the EVO-1 model, which has a custom config class
    ######################
    if hasattr(model, 'config') and not isinstance(model.config, PretrainedConfig):
        delattr(model.config, 'Loader')
        model.config = PretrainedConfig.from_dict(dict(model.config))
    #######################

    target_modules = kwargs.pop("target_modules", None)
    use_rslora = kwargs.pop("use_rslora", True)
    bias = kwargs.pop("bias", "none")
    r = kwargs.pop("r", 32)
    lora_alpha = kwargs.pop("lora_alpha", 256)
    lora_dropout = kwargs.pop("lora_dropout", 0.1)

    if target_modules is None:
        target_modules = find_linear_target_modules(model, keyword_filter=kwargs.get("keyword_filter", None))
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
    def __init__(self, model, **kwargs):
        super(OmniLoraModel, self).__init__()
        target_modules = kwargs.get("target_modules", None)
        if target_modules is None:
            raise ValueError(
                "No target modules found for LoRA injection. To perform LoRA adaptation fine-tuning, "
                "please specify the target modules using the 'target_modules' argument. "
                "The target modules depend on the model architecture, such as 'query', 'value', etc. ")

        self.lora_model = auto_lora_model(model, **kwargs)

        fprint(
            "To reduce GPU memory occupation, "
            "you should avoid include non-trainable parameters into optimizers, "
            "e.g.,  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...), "
            "AVOIDING: optimizer = torch.optim.AdamW(model.parameters(), ...)"
        )

        # self.lora_model is a PEFT wrapped model, which is a wrapper around the original model
        # self.lora_model.base_model is the original omnigenome wrapped model with output heads
        # self.lora_model.base_model.model is the omnigenome model without output heads
        # self.lora_model.base_model.model.model is the transformers-loaded backbone model

        self.config = model.config
        for param in self.parameters():
            self.device = param.device
            self.dtype = param.dtype
            break
        for module in self.lora_model.modules():
            module.device = self.device
            if hasattr(module, 'dtype'):
                module.dtype = self.dtype

        fprint(
            "LoRA model initialized with the following configuration:\n",
            self.lora_model
        )


    def to(self, *args, **kwargs):
        """
        Override the to method to ensure the lora_model is moved to the correct device and dtype.
        """
        self.lora_model.to(*args, **kwargs)
        for param in self.parameters():
            self.device = param.device
            self.dtype = param.dtype
            break
        for module in self.lora_model.modules():
            module.device = self.device
            if hasattr(module, 'dtype'):
                module.dtype = self.dtype
        return self

    def forward(self, *args, **kwargs):
        return self.lora_model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.lora_model.base_model.predict(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.lora_model.base_model.save(*args, **kwargs)

    def model_info(self):
        return self.lora_model.base_model.model_info()

    def set_loss_fn(self, fn):
        return self.lora_model.base_model.set_loss_fn(fn)

    def last_hidden_state_forward(self, **kwargs):
        return self.lora_model.base_model.last_hidden_state_forward(**kwargs)

    def tokenizer(self):
        return self.lora_model.base_model.tokenizer

    def config(self):
        return self.lora_model.base_model.config

    def model(self):
        return self.lora_model.base_model.model