# -*- coding: utf-8 -*-
# file: model_hub.py
# time: 18:13 12/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import json
import os

import autocuda
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from omnigenome.utility.hub_utils import query_models_info, download_model
from ...src.misc.utils import env_meta_info, fprint


class ModelHub:
    """
    A class to handle the loading and management of models from the model hub.
    """
    def __init__(self, *args, **kwargs):
        super(ModelHub, self).__init__(*args, **kwargs)

        self.metadata = env_meta_info()

    @staticmethod
    def load_model_and_tokenizer(
        model_name_or_path,
        local_only=False,
        device=None,
        fast_dtype=torch.float16,
        **kwargs,
    ):
        """
        Loads both the model and tokenizer from a specified model path or hub.

        Args:
            model_name_or_path: The model's name or path.
            local_only: Whether to load only from local sources.
            device: The device to load the model to (if None, auto-selects device).
            fast_dtype: Data type to use (default: float16).
            **kwargs: Additional arguments passed to model and tokenizer loading.

        Returns:
            model, tokenizer: The loaded model and tokenizer.
        """
        model = ModelHub.load(model_name_or_path, local_only=local_only, **kwargs)
        fprint(f"The model and tokenizer has been loaded from {model_name_or_path}.")
        model.to(fast_type=fast_dtype)
        if device is None:
            device = autocuda.auto_cuda()
            fprint(
                f"No device is specified, the model will be loaded to the default device: {device}"
            )
            model.to(device)
        else:
            model.to(device)
        return model, model.tokenizer

    @staticmethod
    def load(
        model_name_or_path,
        local_only=False,
        device=None,
        fast_dtype=torch.float16,
        **kwargs,
    ):
        """
        Loads a model from a specified path or the model hub.

        Args:
            model_name_or_path: The model's name or path.
            local_only: Whether to load only from local sources.
            device: The device to load the model to (if None, auto-selects device).
            fast_dtype: Data type to use (default: float16).
            **kwargs: Additional arguments passed to model loading.

        Returns:
            model: The loaded model.
        """
        if isinstance(model_name_or_path, str) and os.path.exists(model_name_or_path):
            path = model_name_or_path
        elif isinstance(model_name_or_path, str) and not os.path.exists(
            model_name_or_path
        ):
            path = download_model(model_name_or_path, local_only=local_only, **kwargs)
        else:
            raise ValueError("model_name_or_path must be a string.")

        import importlib

        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        with open(f"{path}/metadata.json", "r", encoding="utf8") as f:
            metadata = json.load(f)

        config.metadata = metadata
        base_model = AutoModel.from_config(config, trust_remote_code=True, **kwargs)
        model_lib = importlib.import_module(metadata["library_name"].lower()).model
        model_cls = getattr(model_lib, metadata["model_cls"])

        if "Omni" in metadata["tokenizer_cls"]:
            lib = importlib.import_module(metadata["library_name"].lower())
            tokenizer_cls = getattr(lib, metadata["tokenizer_cls"])
            tokenizer = tokenizer_cls.from_pretrained(path, **kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)

        model = model_cls(
            base_model,
            tokenizer,
            label2id=config.label2id,
            num_labels=config.num_labels,
            **kwargs,
        )
        with open(f"{path}/pytorch_model.bin", "rb") as f:
            model.load_state_dict(
                torch.load(f, map_location=kwargs.get("device", "cpu")), strict=False
            )
        model.to(fast_dtype)
        if device is None:
            device = autocuda.auto_cuda()
            fprint(
                f"No device is specified, the model will be loaded to the default device: {device}"
            )
            model.to(device)
        else:
            model.to(device)
        return model

    def available_models(
        self, model_name_or_path=None, local_only=False, repo="", **kwargs
    ):
        """
        Queries and returns information about available models in the hub.

        Args:
            model_name_or_path: The model's name or path (optional).
            local_only: Whether to load models only from local sources.
            repo: The repository to query models from.
            **kwargs: Additional arguments passed to the query.

        Returns:
            models_info: A list or dictionary with information about the available models.
        """
        models_info = query_models_info(
            model_name_or_path, local_only=local_only, repo=repo, **kwargs
        )
        return models_info

    def push(self, model, **kwargs):
        """
        Pushes a model to the model hub (not implemented).

        Args:
            model: The model to be pushed.
            **kwargs: Additional arguments for pushing the model.
        """
        raise NotImplementedError("This method has not implemented yet.")
