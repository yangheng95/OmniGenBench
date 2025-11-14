# -*- coding: utf-8 -*-
# file: model_hub.py
# time: 14:47 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
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

# Import new HF Hub downloader (with fallback)
try:
    from .hf_download import download_from_hf_hub, verify_download_integrity

    HF_DOWNLOAD_AVAILABLE = True
except ImportError:
    HF_DOWNLOAD_AVAILABLE = False


class GenericOmniModelWrapper(torch.nn.Module):
    """
    Wrapper for standard Transformers models to provide OmniModel-like interface.

    This wrapper adds inference capabilities to standard AutoModel instances
    loaded from HuggingFace Hub, enabling them to work with the OmniGenBench
    inference API even when they weren't originally saved with OmniGenBench.

    Note: This model returns raw logits and hidden states. For task-specific
    inference (e.g., classification with label mapping), use the appropriate
    OmniModelFor* class after fine-tuning on a downstream task.

    Args:
        base_model (torch.nn.Module): The underlying transformer model
        tokenizer (OmniTokenizer): Tokenizer for sequence encoding
    """

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = base_model.config

    def forward(self, **inputs):
        """Forward pass through the base model."""
        return self.base_model(**inputs)

    def __call__(self, **inputs):
        """Allow calling the model directly."""
        return self.forward(**inputs)

    def inference(self, sequence_or_inputs, **kwargs):
        """
        Perform inference on genomic sequences.

        This method provides a basic inference interface for pre-trained models
        that haven't been fine-tuned on a downstream task. It returns raw model
        outputs including logits and hidden states.

        Warning: For meaningful predictions on specific tasks (e.g., classification,
        regression), you should fine-tune the model using AutoTrain or manually
        train with task-specific labels, then load with the appropriate
        OmniModelFor* class.

        Args:
            sequence_or_inputs: Can be:
                - str: Single DNA/RNA sequence
                - list: List of sequences
                - dict: Dictionary with tokenized inputs
            **kwargs: Additional tokenization arguments

        Returns:
            dict: Dictionary containing:
                - 'logits': Raw model outputs (if available)
                - 'last_hidden_state': Final hidden states
                - 'predictions': Same as logits (for API compatibility)
                - 'warning': Message about using pre-trained model

        Example:
            >>> model = ModelHub.load("yangheng/OmniGenome-186M")
            >>> result = model.inference("ATCGATCG")
            >>> print(result.keys())  # ['logits', 'last_hidden_state', 'predictions', 'warning']
        """
        # Tokenize input if needed
        if isinstance(sequence_or_inputs, str):
            inputs = self.tokenizer(
                sequence_or_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                **kwargs,
            )
        elif isinstance(sequence_or_inputs, list):
            inputs = self.tokenizer(
                sequence_or_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                **kwargs,
            )
        elif isinstance(sequence_or_inputs, dict):
            if "input_ids" in sequence_or_inputs:
                # Already tokenized
                inputs = sequence_or_inputs
            else:
                # Assume it's a data dict with 'sequence' field
                seq = sequence_or_inputs.get("sequence") or sequence_or_inputs.get(
                    "seq"
                )
                if seq is None:
                    raise ValueError(
                        "Input dict must contain 'sequence' or 'seq' field"
                    )
                inputs = self.tokenizer(
                    seq, return_tensors="pt", padding=True, truncation=True, **kwargs
                )
        else:
            raise TypeError(f"Unsupported input type: {type(sequence_or_inputs)}")

        # Move to same device as model
        device = next(self.base_model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Run inference
        self.eval()
        with torch.no_grad():
            outputs = self.base_model(**inputs)

        # Build result dictionary
        result = {
            "warning": (
                "You are using a pre-trained model without task-specific fine-tuning. "
                "The outputs are raw logits/embeddings. For meaningful predictions, "
                "please fine-tune this model on a downstream task using AutoTrain or "
                "load a fine-tuned model with the appropriate OmniModelFor* class."
            )
        }

        # Extract logits if available
        if hasattr(outputs, "logits"):
            result["logits"] = outputs.logits
            result["predictions"] = outputs.logits

        # Extract hidden states
        if hasattr(outputs, "last_hidden_state"):
            result["last_hidden_state"] = outputs.last_hidden_state
            # If no logits, use hidden states as predictions for API compatibility
            if "predictions" not in result:
                result["predictions"] = outputs.last_hidden_state

        return result

    def predict(self, sequence_or_inputs, **kwargs):
        """Alias for inference method for API compatibility."""
        return self.inference(sequence_or_inputs, **kwargs)

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.base_model = self.base_model.to(*args, **kwargs)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.base_model.eval()
        return self

    def train(self, mode=True):
        """Set model to training mode."""
        self.base_model.train(mode)
        return self


def download_hf_model(
    model_id,
    cache_dir="__OMNIGENBENCH_DATA__/models/",
    force_download=False,
    use_hf_api=True,
):
    """
    Download a model from Hugging Face Hub to local directory.

    This function provides two download strategies:
    1. HuggingFace Hub API (default, recommended) - No git-lfs required
    2. Git clone (fallback) - Requires git and git-lfs for large files

    Args:
        model_id (str): Hugging Face model identifier (e.g., "yangheng/OmniGenome-186M")
        cache_dir (str): Local directory to store downloaded models
        force_download (bool): Whether to re-download if model already exists locally
        use_hf_api (bool): Whether to use HF Hub API (True) or git clone (False). Defaults to True

    Returns:
        str: Path to the locally downloaded model directory

    Raises:
        subprocess.CalledProcessError: If git clone fails (when use_hf_api=False)
        FileNotFoundError: If git is not available (when use_hf_api=False)
        OSError: If HF Hub download fails (when use_hf_api=True)
    """
    # Strategy 1: Use HuggingFace Hub API (no git-lfs required)
    if use_hf_api and HF_DOWNLOAD_AVAILABLE:
        try:
            fprint(f"Using HuggingFace Hub API to download {model_id}")
            path = download_from_hf_hub(
                repo_id=model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                repo_type="model",
            )
            # Verify download integrity
            if verify_download_integrity(path):
                return path
            else:
                fprint("Download verification failed, will retry with git clone")
                use_hf_api = False  # Fall back to git clone
        except Exception as e:
            fprint(f"HF Hub API download failed: {e}")
            fprint("Falling back to git clone method")
            use_hf_api = False

    # Strategy 2: Use git clone (original method)
    if not use_hf_api or not HF_DOWNLOAD_AVAILABLE:
        if not HF_DOWNLOAD_AVAILABLE:
            fprint("HuggingFace Hub API not available, using git clone")

        return clone_hf_model(model_id, cache_dir, force_download)


def clone_hf_model(
    model_id, cache_dir="__OMNIGENBENCH_DATA__/models/", force_download=False
):
    """
    Clone a model from Hugging Face Hub to local directory using git.

    Note: This method requires git and git-lfs to be installed for large files.
    Consider using download_hf_model(use_hf_api=True) for a more robust approach.

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


def _synchronize_config_with_checkpoint(config, state_dict):
    """
    Align config fields with the actual checkpoint tensor shapes.

    Some training scripts mutate the transformer architecture (e.g., GLU width)
    without updating ``config.json`` before saving. When we later reconstruct
    the model purely from the config we end up with inconsistent layer shapes
    and the checkpoint refuses to load. This helper inspects representative
    weight matrices and overrides the config so the instantiated base model
    matches the saved tensors.
    """

    try:
        # Feed-forward output projection encodes the effective intermediate size.
        ff_key = "model.encoder.layer.0.output.dense.weight"
        if hasattr(config, "intermediate_size") and ff_key in state_dict:
            checkpoint_size = state_dict[ff_key].shape[1]
            if checkpoint_size != config.intermediate_size:
                fprint(
                    "Detected mismatch between config.intermediate_size "
                    f"({config.intermediate_size}) and checkpoint ({checkpoint_size}). "
                    "Overriding config to match checkpoint."
                )
                config.intermediate_size = checkpoint_size

        # Embedding matrix encodes the actual vocab size.
        embed_key = "model.embeddings.word_embeddings.weight"
        if hasattr(config, "vocab_size") and embed_key in state_dict:
            checkpoint_vocab = state_dict[embed_key].shape[0]
            if checkpoint_vocab != config.vocab_size:
                fprint(
                    "Detected mismatch between config.vocab_size "
                    f"({config.vocab_size}) and checkpoint ({checkpoint_vocab}). "
                    "Overriding config to match checkpoint."
                )
                config.vocab_size = checkpoint_vocab
    except Exception as exc:  # pragma: no cover - telemetry only
        fprint(f"Warning: Failed to reconcile config with checkpoint: {exc}")


class ModelHub:
    """
    Centralized hub for loading and managing pre-trained genomic foundation models.

    This class provides a unified interface for model acquisition and loading, implementing
    a hybrid strategy: clone from HuggingFace Hub to local cache on first access, then
    load exclusively from local files for all subsequent operations. This approach ensures:

    - **Reproducibility**: Local caching prevents silent model updates
    - **Offline Access**: Models remain available without internet connectivity after initial clone
    - **Version Control**: Explicit control over model versions via git tags/commits
    - **Large File Handling**: Git LFS support for multi-gigabyte model weights

    **Architecture**: The ModelHub implements a three-tier loading strategy:

    1. **HuggingFace Hub Cloning** (Primary): Uses git to clone model repositories to
       ``__OMNIGENBENCH_DATA__/models/`` directory, preserving full git history and metadata
    2. **OmniGenome Hub Fallback** (Secondary): Legacy download mechanism for models not
       on HuggingFace Hub (deprecated, maintained for backward compatibility)
    3. **Local-Only Loading** (Always): After cloning, all model loading uses ``local_files_only=True``
       to ensure deterministic behavior

    **Model Type Detection**: The hub supports two model categories:

    - **OmniGenBench Models**: Saved via OmniGenBench with ``metadata.json`` containing task
      type, label mappings, and custom attributes. The hub reconstructs the original
      task-specific model class (e.g., OmniModelForSequenceClassification) from metadata.
    - **Standard Transformers Models**: Generic HuggingFace models without OmniGenBench
      metadata. Loaded as base AutoModel instances with attached tokenizer for compatibility.

    **Cache Management**: Models are cached with HuggingFace Hub naming convention:

    - ``yangheng/OmniGenome-186M`` → ``__OMNIGENBENCH_DATA__/models/yangheng--OmniGenome-186M/``
    - Supports ``force_download=True`` to re-clone updated versions
    - No automatic cache cleanup; manual management required for disk space constraints

    Attributes:
        metadata (dict): Framework environment metadata including version information and
            system configuration.

    Example:
        >>> from omnigenbench import ModelHub
        >>> hub = ModelHub()

        >>> # Clone and load a model from HuggingFace Hub (cached for future use)
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer("yangheng/OmniGenome-186M")

        >>> # Load from existing local cache (instant, no network access)
        >>> model, tokenizer = ModelHub.load_model_and_tokenizer("yangheng/OmniGenome-186M")

        >>> # Load only the model (tokenizer already available)
        >>> model = ModelHub.load("yangheng/ogb_tfb_finetuned")

        >>> # Force re-clone to get latest version
        >>> model = ModelHub.load(
        ...     "yangheng/OmniGenome-186M",
        ...     force_download=True
        ... )

        >>> # Load from local directory (no cloning)
        >>> model = ModelHub.load("/path/to/local/model")
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
        config_or_model,
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
            config_or_model (str): Name or path of the model to load.
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
            config_or_model,
            local_only=local_only,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        fprint(f"The model and tokenizer has been loaded from {config_or_model}.")
        return model, model.tokenizer

    @staticmethod
    def load(
        config_or_model,
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
            config_or_model (str): Name or path of the model to load.
                Can be a local path, Hugging Face model identifier (e.g., "yangheng/OmniGenome-186M"),
                or a model name from the OmniGenome hub (fallback).
            local_only (bool, optional): Whether to use only local cache. Defaults to False
            device (str, optional): Device to load the model on. If None, uses auto-detection
            dtype (torch.dtype, optional): Data type for the model. Defaults to torch.float16
            **kwargs: Additional keyword arguments:
                - cache_dir (str): Directory to store cloned models. Defaults to "__OMNIGENBENCH_DATA__/models/"
                - force_download (bool): Whether to re-clone even if model exists locally. Defaults to False

        Returns:
            torch.nn.Module: The loaded model

        Raises:
            ValueError: If config_or_model is not a string
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
        if not isinstance(config_or_model, str):
            raise ValueError("config_or_model must be a string.")

        # Determine the model path - always ensure we have a local path
        if os.path.exists(config_or_model):
            # Local path exists
            path = config_or_model
            fprint(f"Using existing local model path: {path}")
        else:
            # Check if it looks like a Hugging Face model identifier (contains "/" or is a known model name)
            if "/" in config_or_model or config_or_model in [
                "bert-base-uncased",
                "gpt2",
                "roberta-base",
                "distilbert-base-uncased",
            ]:
                # Download from Hugging Face Hub to local directory
                # Prioritize HF Hub API (no git-lfs) over git clone
                try:
                    cache_dir = kwargs.pop("cache_dir", "__OMNIGENBENCH_DATA__/models/")
                    force_download = kwargs.pop("force_download", False)
                    use_hf_api = kwargs.pop("use_hf_api", True)  # Default to HF API

                    path = download_hf_model(
                        config_or_model,
                        cache_dir,
                        force_download,
                        use_hf_api=use_hf_api,
                    )
                    fprint(f"Downloaded model from Hugging Face Hub to: {path}")
                except Exception as hf_error:
                    # Fallback to OmniGenome hub download
                    fprint(
                        f"Failed to download from HF Hub ({hf_error}), trying OmniGenome hub..."
                    )
                    try:
                        path = download_model(
                            config_or_model, local_only=local_only, **kwargs
                        )
                    except Exception as og_error:
                        raise FileNotFoundError(
                            f"Could not load model '{config_or_model}' from either "
                            f"Hugging Face Hub or OmniGenome hub. "
                            f"HF error: {hf_error}. OG error: {og_error}"
                        )
            else:
                # Try OmniGenome hub first for non-HF identifiers
                try:
                    path = download_model(
                        config_or_model, local_only=local_only, **kwargs
                    )
                    fprint(f"Downloaded model from OmniGenome hub to: {path}")
                except Exception as og_error:
                    raise FileNotFoundError(
                        f"Could not find model '{config_or_model}' in OmniGenome hub. "
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
                state_dict = None
                state_dict_path = os.path.join(path, "pytorch_model.bin")
                safetensors_path = os.path.join(path, "model.safetensors")

                try:
                    if os.path.exists(state_dict_path):
                        fprint(f"Reading checkpoint tensors from: {state_dict_path}")
                        state_dict = torch.load(state_dict_path, map_location="cpu")
                    elif os.path.exists(safetensors_path):
                        fprint(
                            f"Reading checkpoint tensors from safetensors: {safetensors_path}"
                        )
                        from safetensors.torch import load_file

                        state_dict = load_file(safetensors_path)
                    else:
                        fprint(
                            "Warning: No pytorch_model.bin or model.safetensors found"
                        )
                except Exception as exc:
                    fprint(f"Warning: Could not read checkpoint tensors: {exc}")

                if state_dict is not None:
                    _synchronize_config_with_checkpoint(config, state_dict)

                architecture_cls = None
                if getattr(config, "architectures", None):
                    architecture_cls = config.architectures[0]

                base_model = None
                module_candidates = []
                auto_map = getattr(config, "auto_map", {}) or {}
                for target in auto_map.values():
                    if isinstance(target, str) and "." in target:
                        module_candidates.append(target.split(".")[0])

                # Preserve order but drop duplicates
                seen = set()
                module_candidates = [
                    m for m in module_candidates if not (m in seen or seen.add(m))
                ]

                for module_name in module_candidates:
                    candidate_file = os.path.join(path, f"{module_name}.py")
                    if not os.path.exists(candidate_file):
                        continue
                    if architecture_cls is None:
                        continue
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, candidate_file
                        )
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        if hasattr(module, architecture_cls):
                            base_model_cls = getattr(module, architecture_cls)
                            base_model = base_model_cls(config)
                            fprint(
                                f"Instantiated base architecture {architecture_cls} "
                                f"from local module {module_name}.py"
                            )
                            break
                    except Exception as exc:
                        fprint(
                            f"Warning: Failed to instantiate {architecture_cls} from {module_name}.py: {exc}"
                        )

                if base_model is None:
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

                # Initialize the OmniModel subclass directly from the in-memory base model
                # to avoid re-loading checkpoints with potentially incompatible HF versions.
                model = model_cls(
                    base_model,
                    tokenizer,
                    **init_kwargs,
                )

                if state_dict is not None:
                    try:
                        model.load_state_dict(state_dict, strict=False)
                    except Exception as exc:
                        fprint(f"Warning: Could not load state dict: {exc}")
            else:
                # Fallback to standard Transformers model
                fprint(
                    f"Could not load custom model class {metadata['model_cls']}, falling back to standard model"
                )
                base_model = AutoModel.from_pretrained(
                    path,
                    config=config,
                    trust_remote_code=True,
                    local_files_only=True,
                    **kwargs,
                )
                # Wrap with GenericOmniModelWrapper to provide inference interface
                model = GenericOmniModelWrapper(base_model, tokenizer)
                model.tokenizer = tokenizer
        else:
            # Standard Transformers model - load from local path
            fprint("Loading as standard Transformers model from local path")
            base_model = AutoModel.from_pretrained(
                path,
                config=config,
                trust_remote_code=True,
                local_files_only=True,  # Force local files only
                **kwargs,
            )
            # Wrap with GenericOmniModelWrapper to provide inference interface
            model = GenericOmniModelWrapper(base_model, tokenizer)
            model.tokenizer = tokenizer

        if isinstance(dtype, str):
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is None:
                raise ValueError(f"Unsupported dtype string '{dtype}'.")
            dtype = torch_dtype

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
        self, config_or_model=None, local_only=False, repo="", **kwargs
    ):
        """
        Get information about available models in the hub.

        This method queries the OmniGenome hub to retrieve information about
        available models. It can filter models by name and supports both
        local and remote queries.

        Args:
            config_or_model (str, optional): Filter models by name. Defaults to None
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
            config_or_model, local_only=local_only, repo=repo, **kwargs
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
