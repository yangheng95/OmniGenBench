# -*- coding: utf-8 -*-
# file: hf_download.py
# time: 15:30 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Robust HuggingFace Hub downloader without git-lfs dependency.

This module provides reliable model/dataset downloading using the official
HuggingFace Hub API, eliminating the need for git and git-lfs.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Union
import warnings

try:
    from huggingface_hub import (
        snapshot_download,
        hf_hub_download,
        HfApi,
        list_repo_files,
    )

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    warnings.warn(
        "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
    )

from omnigenbench.src.misc.utils import fprint


def download_from_hf_hub(
    repo_id: str,
    cache_dir: str = "__OMNIGENOME_DATA__/models/",
    force_download: bool = False,
    repo_type: str = "model",
    allow_patterns: Optional[Union[str, List[str]]] = None,
    ignore_patterns: Optional[Union[str, List[str]]] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download model or dataset from HuggingFace Hub using official API (no git-lfs required).

    This function uses HuggingFace's `snapshot_download` which handles large files
    efficiently through their CDN without requiring git-lfs installation.

    Args:
        repo_id (str): HuggingFace repository identifier (e.g., "yangheng/OmniGenome-186M")
        cache_dir (str): Local directory to cache downloads. Defaults to "__OMNIGENOME_DATA__/models/"
        force_download (bool): Whether to re-download even if files exist. Defaults to False
        repo_type (str): Type of repository - "model", "dataset", or "space". Defaults to "model"
        allow_patterns (Optional[Union[str, List[str]]]): Patterns to include (e.g., ["*.json", "*.bin"])
        ignore_patterns (Optional[Union[str, List[str]]]): Patterns to exclude (e.g., ["*.msgpack"])
        token (Optional[str]): HuggingFace API token for private repos

    Returns:
        str: Path to the downloaded repository

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If repository cannot be accessed
        OSError: If download fails

    Example:
        >>> # Download a model
        >>> path = download_from_hf_hub("yangheng/OmniGenome-186M")
        >>> print(f"Model downloaded to: {path}")

        >>> # Download only specific files
        >>> path = download_from_hf_hub(
        ...     "yangheng/ogb_tfb_finetuned",
        ...     allow_patterns=["*.json", "*.bin", "*.txt"]
        ... )

        >>> # Force re-download
        >>> path = download_from_hf_hub(
        ...     "yangheng/OmniGenome-186M",
        ...     force_download=True
        ... )

    Note:
        - This method does NOT require git or git-lfs
        - Files are downloaded via HTTPS from HuggingFace's CDN
        - Large files are automatically chunked and verified
        - Uses HuggingFace's native caching mechanism with symlinks
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required for this download method. "
            "Install it with: pip install huggingface_hub"
        )

    # Create cache directory if needed
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Convert repo_id to safe directory name for our custom cache structure
    safe_model_name = repo_id.replace("/", "--")
    local_model_path = cache_path / safe_model_name

    # Check if already downloaded and not forcing re-download
    if local_model_path.exists() and not force_download:
        fprint(f"[INFO] Repository {repo_id} already exists at {local_model_path}")
        return str(local_model_path)

    # Remove existing directory if force_download
    if local_model_path.exists() and force_download:
        fprint(
            f"[INFO] Removing existing directory for re-download: {local_model_path}"
        )
        shutil.rmtree(local_model_path)

    fprint(f"[INFO] Downloading {repo_type} '{repo_id}' from HuggingFace Hub...")
    fprint("[INFO] This may take a while for large models (no git-lfs required)")

    try:
        # Use snapshot_download with custom cache location
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_path),
            local_dir=str(local_model_path),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            repo_type=repo_type,
            force_download=force_download,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=token,
        )

        fprint(f"[SUCCESS] Successfully downloaded {repo_id} to {downloaded_path}")
        return str(downloaded_path)

    except Exception as e:
        fprint(f"[ERROR] Failed to download {repo_id}: {str(e)}")
        # Clean up partial download
        if local_model_path.exists():
            fprint(f"[INFO] Cleaning up partial download at {local_model_path}")
            shutil.rmtree(local_model_path)
        raise


def download_file_from_hf_hub(
    repo_id: str,
    filename: str,
    cache_dir: str = "__OMNIGENOME_DATA__/models/",
    force_download: bool = False,
    repo_type: str = "model",
    token: Optional[str] = None,
) -> str:
    """
    Download a single file from HuggingFace Hub.

    Args:
        repo_id (str): HuggingFace repository identifier
        filename (str): Name of file to download (e.g., "pytorch_model.bin")
        cache_dir (str): Local directory to cache downloads
        force_download (bool): Whether to re-download even if file exists
        repo_type (str): Type of repository - "model", "dataset", or "space"
        token (Optional[str]): HuggingFace API token for private repos

    Returns:
        str: Path to the downloaded file

    Example:
        >>> # Download specific model file
        >>> path = download_file_from_hf_hub(
        ...     "yangheng/ogb_tfb_finetuned",
        ...     "pytorch_model.bin"
        ... )
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fprint(f"[INFO] Downloading {filename} from {repo_id}...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_path),
            force_download=force_download,
            repo_type=repo_type,
            token=token,
        )

        fprint(f"[SUCCESS] Downloaded {filename} to {downloaded_path}")
        return downloaded_path

    except Exception as e:
        fprint(f"[ERROR] Failed to download {filename}: {str(e)}")
        raise


def list_hf_repo_files(
    repo_id: str,
    repo_type: str = "model",
    token: Optional[str] = None,
) -> List[str]:
    """
    List all files in a HuggingFace repository.

    Args:
        repo_id (str): HuggingFace repository identifier
        repo_type (str): Type of repository - "model", "dataset", or "space"
        token (Optional[str]): HuggingFace API token for private repos

    Returns:
        List[str]: List of file paths in the repository

    Example:
        >>> files = list_hf_repo_files("yangheng/OmniGenome-186M")
        >>> print(files)
        ['config.json', 'pytorch_model.bin', 'tokenizer.json', ...]
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    try:
        api = HfApi()
        files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
        return files
    except Exception as e:
        fprint(f"[ERROR] Failed to list files in {repo_id}: {str(e)}")
        raise


def verify_download_integrity(
    local_path: str, required_files: Optional[List[str]] = None
) -> bool:
    """
    Verify that a downloaded repository has all required files.

    Args:
        local_path (str): Path to downloaded repository
        required_files (Optional[List[str]]): List of required files to check.
            If None, checks for common model files.

    Returns:
        bool: True if all required files exist

    Example:
        >>> is_valid = verify_download_integrity(
        ...     "__OMNIGENOME_DATA__/models/yangheng--OmniGenome-186M",
        ...     required_files=["config.json", "pytorch_model.bin"]
        ... )
    """
    local_path = Path(local_path)

    if not local_path.exists():
        fprint(f"[ERROR] Path does not exist: {local_path}")
        return False

    # Default required files for models
    if required_files is None:
        required_files = ["config.json"]  # Minimal requirement

    missing_files = []
    for filename in required_files:
        file_path = local_path / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        fprint(f"[WARNING] Missing files in {local_path}: {missing_files}")
        return False

    # Check for LFS pointer files (indicates incomplete git-lfs download)
    for file_path in local_path.rglob("*.bin"):
        if file_path.stat().st_size < 200:  # Suspiciously small
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                if "version https://git-lfs" in first_line:
                    fprint(
                        f"[ERROR] Detected git-lfs pointer file (incomplete download): {file_path}"
                    )
                    fprint(
                        "[ERROR] Please use download_from_hf_hub() to download properly"
                    )
                    return False

    fprint(f"[SUCCESS] All required files present in {local_path}")
    return True


def get_model_info(repo_id: str, token: Optional[str] = None) -> dict:
    """
    Get metadata about a HuggingFace model repository.

    Args:
        repo_id (str): HuggingFace repository identifier
        token (Optional[str]): HuggingFace API token for private repos

    Returns:
        dict: Repository metadata including size, last modified, etc.

    Example:
        >>> info = get_model_info("yangheng/OmniGenome-186M")
        >>> print(f"Model size: {info.get('size', 'unknown')}")
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )

    try:
        api = HfApi()
        model_info = api.model_info(repo_id=repo_id, token=token)
        return {
            "id": model_info.id,
            "sha": model_info.sha,
            "last_modified": (
                str(model_info.lastModified) if model_info.lastModified else None
            ),
            "tags": model_info.tags,
            "pipeline_tag": model_info.pipeline_tag,
            "siblings": (
                [f.rfilename for f in model_info.siblings]
                if model_info.siblings
                else []
            ),
        }
    except Exception as e:
        fprint(f"[ERROR] Failed to get info for {repo_id}: {str(e)}")
        raise
