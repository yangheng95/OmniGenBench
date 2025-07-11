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
import dill
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from omnigenome.utility.hub_utils import query_models_info, download_model
from ...src.misc.utils import env_meta_info, fprint


class ModelHub:
    """
    A hub for loading and managing pre-trained genomic models.
    
    This class provides a unified interface for loading pre-trained models
    from the OmniGenome hub or local paths. It handles model downloading,
    tokenizer loading, and device placement automatically.
    
    The ModelHub supports various model types and can automatically
    download models from the hub if they're not available locally.
    
    Attributes:
        metadata (dict): Environment metadata information
        
    Example:
        >>> from omnigenome import ModelHub
        >>> hub = ModelHub()
        
        >>> # Load a model from the hub
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer("model_name")
        
        >>> # Check available models
        >>> models = hub.available_models()
        >>> print(list(models.keys()))
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the ModelHub instance.
        
        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super(ModelHub, self).__init__(*args, **kwargs)

        self.metadata = env_meta_info()

    @staticmethod
    def load_model_and_tokenizer(
        model_name_or_path,
        local_only=False,
        device=None,
        dtype=torch.float16,
        **kwargs,
    ):
        """
        Load a model and its tokenizer from the hub or local path.
        
        This method loads both the model and tokenizer, places them on the
        specified device, and returns them as a tuple. It handles automatic
        device selection if none is specified.
        
        Args:
            model_name_or_path (str): Name or path of the model to load
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            device (str, optional): Device to load the model on. If None, uses auto-detection
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16
            **kwargs: Additional keyword arguments passed to the model loading functions
            
        Returns:
            tuple: A tuple containing (model, tokenizer)
            
        Example:
            >>> model, tokenizer = ModelHub.load_model_and_tokenizer("yangheng/OmniGenome-186M")
            >>> print(f"Model loaded on device: {next(model.parameters()).device}")
        """
        model = ModelHub.load(model_name_or_path, local_only=local_only, **kwargs)
        fprint(f"The model and tokenizer has been loaded from {model_name_or_path}.")
        model.to(dtype)
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
        dtype=torch.float16,
        **kwargs,
    ):
        """
        Load a model from the hub or local path.
        
        This method handles model loading from various sources including
        local paths and the OmniGenome hub. It automatically downloads
        models if they're not available locally.
        
        Args:
            model_name_or_path (str): Name or path of the model to load
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            device (str, optional): Device to load the model on. If None, uses auto-detection
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16
            **kwargs: Additional keyword arguments passed to the model loading functions
            
        Returns:
            torch.nn.Module: The loaded model
            
        Raises:
            ValueError: If model_name_or_path is not a string
            
        Example:
            >>> model = ModelHub.load("yangheng/OmniGenome-186M")
            >>> print(f"Model type: {type(model)}")
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

        if "Omni" in metadata["tokenizer_cls"]:
            lib = importlib.import_module(metadata["library_name"].lower())
            tokenizer_cls = getattr(lib, metadata["tokenizer_cls"])
            tokenizer = tokenizer_cls.from_pretrained(path, **kwargs)
        else:
            from multimolecule import RnaTokenizer
            tokenizer = RnaTokenizer.from_pretrained(path, **kwargs)

        config.metadata = metadata

        base_model = AutoModel.from_config(config, trust_remote_code=True, **kwargs)
        model_lib = importlib.import_module(metadata["library_name"].lower()).model
        model_cls = getattr(model_lib, metadata["model_cls"])
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
        model.to(dtype)
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
        Get information about available models in the hub.
        
        This method queries the OmniGenome hub to retrieve information about
        available models. It can filter models by name and supports both
        local and remote queries.
        
        Args:
            model_name_or_path (str, optional): Filter models by name. Defaults to None
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            repo (str, optional): Repository URL to query. Defaults to ""
            **kwargs: Additional keyword arguments
            
        Returns:
            dict: Dictionary containing information about available models
            
        Example:
            >>> hub = ModelHub()
            >>> models = hub.available_models()
            >>> print(f"Available models: {len(models)}")
            
            >>> # Filter models by name
            >>> dna_models = hub.available_models("DNA")
            >>> print(f"DNA models: {list(dna_models.keys())}")
        """
        models_info = query_models_info(
            model_name_or_path, local_only=local_only, repo=repo, **kwargs
        )
        return models_info

    def push(self, model, **kwargs):
        """
        Push a model to the hub.
        
        This method is not yet implemented and will raise a NotImplementedError.
        
        Args:
            model: The model to push to the hub
            **kwargs: Additional keyword arguments
            
        Raises:
            NotImplementedError: This method has not been implemented yet
        """
        raise NotImplementedError("This method has not implemented yet.")
