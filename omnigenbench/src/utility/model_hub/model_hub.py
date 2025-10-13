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
import subprocess
import shutil
import sys
import importlib.util
from pathlib import Path

import autocuda
import torch
from transformers import AutoConfig, AutoModel

from ..hub_utils import query_models_info, download_model
from omnigenbench.src.misc.utils import env_meta_info, fprint
from omnigenbench.src.abc.abstract_tokenizer import OmniTokenizer


def clone_hf_model(
    model_id, cache_dir="__OMNIGENOME_DATA__/models/", force_download=False
):
    """
    Clone a model from Hugging Face Hub to local directory using git.

    Args:
        model_id (str): Hugging Face model identifier (e.g., "yangheng/OmniGenome-186M")
        cache_dir (str): Local directory to store cloned models
        force_download (bool): Whether to re-download if model already exists locally

    Returns:
        str: Path to the locally cloned model directory

    Raises:
        subprocess.CalledProcessError: If git clone fails
        FileNotFoundError: If git is not available
    """
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Convert model_id to safe directory name
    safe_model_name = model_id.replace("/", "--")
    local_model_path = cache_path / safe_model_name

    # Check if model already exists locally
    if local_model_path.exists() and not force_download:
        fprint(f"Model {model_id} already exists locally at {local_model_path}")
        return str(local_model_path)

    # Remove existing directory if force_download is True
    if local_model_path.exists() and force_download:
        fprint(f"Removing existing model directory: {local_model_path}")
        shutil.rmtree(local_model_path)

    # Clone the model repository
    hf_repo_url = f"https://huggingface.co/{model_id}"
    fprint(f"Cloning model {model_id} from {hf_repo_url}...")

    try:
        # Check if git-lfs is available
        subprocess.run(
            ["git", "lfs", "version"], capture_output=True, check=True, text=True
        )
        fprint("Git LFS is available for handling large files")
    except (subprocess.CalledProcessError, FileNotFoundError):
        fprint(
            "Warning: Git LFS not found. Large model files may not download correctly."
        )

    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", hf_repo_url, str(local_model_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        fprint(f"Successfully cloned {model_id} to {local_model_path}")
        return str(local_model_path)

    except subprocess.CalledProcessError as e:
        fprint(f"Failed to clone model {model_id}: {e.stderr}")
        # Clean up partial clone if it exists
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
        raise
    except FileNotFoundError:
        raise FileNotFoundError(
            "Git is not available. Please install Git to clone models from Hugging Face Hub."
        )


class ModelHub:
    """
    A hub for loading and managing pre-trained genomic models.

    This class provides a unified interface for loading pre-trained models
    by cloning them from Hugging Face Hub to local directories, loading from
    existing local paths, or downloading from the OmniGenome hub (as fallback).
    All model loading is performed using local file system access only.

    The ModelHub supports various model types including standard Transformers models
    and OmniGenome models with custom metadata. It prioritizes cloning from
    Hugging Face Hub to ensure local access and falls back to the OmniGenome hub if needed.

    Key Features:
    - Clones models from HF Hub using git (with Git LFS support for large files)
    - Caches models locally to avoid repeated downloads
    - Loads all models from local file system only (no online transformers loading)
    - Supports both OmniGenome models and standard Transformers models

    Attributes:
        metadata (dict): Environment metadata information

    Example:
        >>> from omnigenbench import ModelHub
        >>> hub = ModelHub()

        >>> # Clone and load a model from Hugging Face Hub
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer("yangheng/OmniGenome-186M")

        >>> # Load a model from local path
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer("/path/to/local/model")

        >>> # Force re-download/clone of a model
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer(
        ...     "yangheng/OmniGenome-186M",
        ...     force_download=True
        ... )
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
        Load a model and its tokenizer from Hugging Face Hub or local path.

        This method loads both the model and tokenizer, places them on the
        specified device, and returns them as a tuple. It handles automatic
        device selection if none is specified.

        Args:
            model_name_or_path (str): Name or path of the model to load.
                Can be a local path, Hugging Face model identifier (e.g., "yangheng/OmniGenome-186M"),
                or a model name from the OmniGenome hub (fallback).
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            device (str, optional): Device to load the model on. If None, uses auto-detection
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16
            **kwargs: Additional keyword arguments passed to the model loading functions

        Returns:
            tuple: A tuple containing (model, tokenizer)

        Example:
            >>> # Load from Hugging Face Hub
            >>> model, tokenizer = ModelHub.load_model_and_tokenizer("yangheng/OmniGenome-186M")
            >>> # Load from local path
            >>> model, tokenizer = ModelHub.load_model_and_tokenizer("/path/to/local/model")
            >>> # Force re-download
            >>> model, tokenizer = ModelHub.load_model_and_tokenizer(
            ...     "yangheng/OmniGenome-186M",
            ...     force_download=True
            ... )
        """
        model = ModelHub.load(
            model_name_or_path,
            local_only=local_only,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        fprint(f"The model and tokenizer has been loaded from {model_name_or_path}.")
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
        Load a model by cloning from Hugging Face Hub or using local path.

        This method handles model loading by first cloning models from Hugging Face Hub
        to local directories, then loading from the local file system. It supports both
        OmniGenome models with metadata and standard Transformers models. All model
        loading is performed using local files only.

        Args:
            model_name_or_path (str): Name or path of the model to load.
                Can be a local path, Hugging Face model identifier (e.g., "yangheng/OmniGenome-186M"),
                or a model name from the OmniGenome hub (fallback).
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            device (str, optional): Device to load the model on. If None, uses auto-detection
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16
            **kwargs: Additional keyword arguments:
                - cache_dir (str): Directory to store cloned models. Defaults to "__OMNIGENOME_DATA__/models/"
                - force_download (bool): Whether to re-clone even if model exists locally. Defaults to False

        Returns:
            torch.nn.Module: The loaded model

        Raises:
            ValueError: If model_name_or_path is not a string
            FileNotFoundError: If the model cannot be found locally or remotely
            subprocess.CalledProcessError: If git clone fails

        Example:
            >>> # Clone and load from Hugging Face Hub
            >>> model = ModelHub.load("yangheng/OmniGenome-186M")
            >>> # Load from local path
            >>> model = ModelHub.load("/path/to/local/model")
            >>> # Force re-clone
            >>> model = ModelHub.load("yangheng/OmniGenome-186M", force_download=True)
            >>> print(f"Model type: {type(model)}")
        """
        if not isinstance(model_name_or_path, str):
            raise ValueError("model_name_or_path must be a string.")

        # Determine the model path - always ensure we have a local path
        if os.path.exists(model_name_or_path):
            # Local path exists
            path = model_name_or_path
            fprint(f"Using existing local model path: {path}")
        else:
            # Check if it looks like a Hugging Face model identifier (contains "/" or is a known model name)
            if "/" in model_name_or_path or model_name_or_path in [
                "bert-base-uncased",
                "gpt2",
                "roberta-base",
                "distilbert-base-uncased",
            ]:
                # Clone from Hugging Face Hub to local directory
                try:
                    cache_dir = kwargs.get("cache_dir", "__OMNIGENOME_DATA__/models/")
                    force_download = kwargs.get("force_download", False)
                    path = clone_hf_model(model_name_or_path, cache_dir, force_download)
                    fprint(f"Cloned model from Hugging Face Hub to: {path}")
                except Exception as hf_error:
                    # Fallback to OmniGenome hub download
                    fprint(
                        f"Failed to clone from HF Hub ({hf_error}), trying OmniGenome hub..."
                    )
                    try:
                        path = download_model(
                            model_name_or_path, local_only=local_only, **kwargs
                        )
                    except Exception as og_error:
                        raise FileNotFoundError(
                            f"Could not load model '{model_name_or_path}' from either "
                            f"Hugging Face Hub or OmniGenome hub. "
                            f"HF error: {hf_error}. OG error: {og_error}"
                        )
            else:
                # Try OmniGenome hub first for non-HF identifiers
                try:
                    path = download_model(
                        model_name_or_path, local_only=local_only, **kwargs
                    )
                    fprint(f"Downloaded model from OmniGenome hub to: {path}")
                except Exception as og_error:
                    raise FileNotFoundError(
                        f"Could not find model '{model_name_or_path}' in OmniGenome hub. "
                        f"Error: {og_error}"
                    )

        # Load configuration
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, **kwargs)

        # Try to load metadata (for OmniGenome models) - always from local path now
        metadata = None
        metadata_path = os.path.join(path, "metadata.json")
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf8") as f:
                    metadata = json.load(f)
                fprint(f"Loaded metadata from: {metadata_path}")
            else:
                fprint(
                    "No metadata.json found, treating as standard Transformers model"
                )
        except Exception as e:
            fprint(
                f"Could not load metadata.json: {e}, treating as standard Transformers model"
            )

        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(path, **kwargs)

        # Set metadata in config if available
        if metadata:
            config.metadata = metadata

        # Load model based on whether we have metadata - all from local files now
        if metadata and "model_cls" in metadata:
            # Try to load custom model class
            model_cls = None

            # Method 1: Try to load from built-in omnigenbench/omnigenome modules
            if "library_name" in metadata:
                try:
                    import importlib

                    model_lib = importlib.import_module(
                        metadata["library_name"].lower()
                    ).model
                    model_cls = getattr(model_lib, metadata["model_cls"])
                    fprint(
                        f"Loaded model class {metadata['model_cls']} from {metadata['library_name']}"
                    )
                except (ImportError, AttributeError) as e:
                    fprint(f"Could not load from library_name: {e}")

            # Method 2: Try to load from model_module if specified
            if model_cls is None and "model_module" in metadata:
                try:
                    import importlib

                    model_module = importlib.import_module(metadata["model_module"])
                    model_cls = getattr(model_module, metadata["model_cls"])
                    fprint(
                        f"Loaded model class {metadata['model_cls']} from module {metadata['model_module']}"
                    )
                except (ImportError, AttributeError) as e:
                    fprint(f"Could not load from model_module: {e}")

            # Method 3: Try to load from custom_model.py file
            if model_cls is None and "custom_model_file" in metadata:
                custom_model_path = os.path.join(path, metadata["custom_model_file"])
                if os.path.exists(custom_model_path):
                    try:
                        # Dynamically import the custom model file
                        spec = importlib.util.spec_from_file_location(
                            "custom_model", custom_model_path
                        )
                        custom_module = importlib.util.module_from_spec(spec)
                        sys.modules["custom_model"] = custom_module
                        spec.loader.exec_module(custom_module)
                        model_cls = getattr(custom_module, metadata["model_cls"])
                        fprint(
                            f"Loaded custom model class {metadata['model_cls']} from {custom_model_path}"
                        )
                    except Exception as e:
                        fprint(f"Could not load from custom_model_file: {e}")

            # If we successfully loaded the model class, instantiate it
            if model_cls is not None:
                base_model = AutoModel.from_config(
                    config, trust_remote_code=True, **kwargs
                )

                # Prepare initialization parameters
                init_kwargs = {
                    "label2id": getattr(config, "label2id", {}),
                    "num_labels": getattr(config, "num_labels", 2),
                }

                # Add custom attributes from metadata
                if "custom_attrs" in metadata:
                    init_kwargs.update(metadata["custom_attrs"])

                # Merge with user-provided kwargs
                init_kwargs.update(kwargs)

                model = model_cls(
                    base_model,
                    tokenizer,
                    **init_kwargs,
                )

                # Load state dict from local files
                state_dict_path = os.path.join(path, "pytorch_model.bin")
                safetensors_path = os.path.join(path, "model.safetensors")

                try:
                    if os.path.exists(state_dict_path):
                        fprint(f"Loading state dict from: {state_dict_path}")
                        with open(state_dict_path, "rb") as f:
                            model.load_state_dict(
                                torch.load(f, map_location=kwargs.get("device", "cpu")),
                                strict=False,
                            )
                    elif os.path.exists(safetensors_path):
                        fprint(
                            f"Loading state dict from safetensors: {safetensors_path}"
                        )
                        from safetensors.torch import load_file

                        state_dict = load_file(safetensors_path)
                        model.load_state_dict(state_dict, strict=False)
                    else:
                        fprint(
                            "Warning: No pytorch_model.bin or model.safetensors found"
                        )
                except Exception as e:
                    fprint(f"Warning: Could not load state dict: {e}")
            else:
                # Fallback to standard Transformers model
                fprint(
                    f"Could not load custom model class {metadata['model_cls']}, falling back to standard model"
                )
                model = AutoModel.from_pretrained(
                    path,
                    config=config,
                    trust_remote_code=True,
                    local_files_only=True,
                    **kwargs,
                )
                model.tokenizer = tokenizer
        else:
            # Standard Transformers model - load from local path
            fprint("Loading as standard Transformers model from local path")
            model = AutoModel.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                local_files_only=True,  # Force local files only
                **kwargs,
            )
            # Attach tokenizer for compatibility
            model.tokenizer = tokenizer

        model.to(dtype)
        if device is None:
            device = autocuda.auto_cuda()
            fprint(
                f"No device is specified, the model will be loaded to the default device: {device}"
            )
            model.to(device)
        else:
            model.to(device)
        model.eval()
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
