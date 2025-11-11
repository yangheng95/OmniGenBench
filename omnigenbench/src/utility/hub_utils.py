# -*- coding: utf-8 -*-
# file: hub_utils.py
# time: 16:54 13/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import json
import os
from typing import Union, Dict, Any

import findfile
import requests
import tqdm
from packaging.version import Version
from termcolor import colored

from omnigenbench import __version__ as current_version
from ...src.misc.utils import fprint, default_omnigenbench_hub_repo


def unzip_checkpoint(checkpoint_path):
    """
    This function extracts a zipped checkpoint file to a directory,
    making it ready for use by the model loading functions.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        str: The path to the extracted checkpoint directory.

    Example:
        >>> extracted_path = unzip_checkpoint("model.zip")
        >>> print(extracted_path)  # "model"
    """
    if not checkpoint_path.endswith(".zip"):
        fprint("Checkpoint path does not end with .zip, returning the original path.")
        return checkpoint_path

    import zipfile

    extract_dir = checkpoint_path.strip(".zip")
    fprint("Unzipping checkpoint from {}...".format(checkpoint_path))
    with zipfile.ZipFile(checkpoint_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def query_models_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    This function retrieves model information from the OmniGenome hub,
    either from a remote repository or from a local cache. It supports
    filtering by keywords to find specific models.

    Args:
        keyword (Union[list, str]): A keyword or list of keywords to filter models.
        repo (str, optional): The repository URL to query. If None, uses the default hub.
        local_only (bool): Whether to use only local cache. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Any]: A dictionary containing model information filtered by the keyword.

    Example:
        >>> # Query all models
        >>> models = query_models_info("")
        >>> print(len(models))  # Number of available models
        >>> # Query specific models
        >>> models = query_models_info("DNA")
        >>> print(models.keys())  # Models containing "DNA"
    """
    if local_only:
        with open("./models_info.json", "r", encoding="utf8") as f:
            models_info = json.load(f)
    else:
        hub_repo = (repo if repo else default_omnigenbench_hub_repo) + "resolve/main/"
        try:
            response = requests.get(hub_repo + "models/models_info.json")
            models_info = response.json()
            with open("./models_info.json", "w", encoding="utf8") as f:
                json.dump(models_info, f)
        except Exception as e:
            fprint(
                "Fail to download models info from OmniGenBench Hub, the error is: {}".format(
                    e
                )
            )
            try:
                with open("./models_info.json", "r", encoding="utf8") as f:
                    models_info = json.load(f)
            except FileNotFoundError:
                fprint("No local models_info.json found, returning empty dict.")
                models_info = {}

    if isinstance(keyword, str):
        filtered_models_info = {}
        for key in models_info:
            if keyword in key:
                filtered_models_info[key] = models_info[key]
        return filtered_models_info
    else:
        return models_info


def query_pipelines_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    This function retrieves pipeline information from the OmniGenome hub,
    either from a remote repository or from a local cache. It supports
    filtering by keywords to find specific pipelines.

    Args:
        keyword (Union[list, str]): A keyword or list of keywords to filter pipelines.
        repo (str, optional): The repository URL to query. If None, uses the default hub.
        local_only (bool): Whether to use only local cache. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Any]: A dictionary containing pipeline information filtered by the keyword.

    Example:
        >>> # Query all pipelines
        >>> pipelines = query_pipelines_info("")
        >>> print(len(pipelines))  # Number of available pipelines
        >>> # Query specific pipelines
        >>> pipelines = query_pipelines_info("classification")
        >>> print(pipelines.keys())  # Pipelines containing "classification"
    """
    if local_only:
        with open("./pipelines_info.json", "r", encoding="utf8") as f:
            pipelines_info = json.load(f)
    else:
        hub_repo = (repo if repo else default_omnigenbench_hub_repo) + "resolve/main/"
        try:
            response = requests.get(hub_repo + "pipelines/pipelines_info.json")
            pipelines_info = response.json()
            with open("./pipelines_info.json", "w", encoding="utf8") as f:
                json.dump(pipelines_info, f)
        except Exception as e:
            fprint(
                "Fail to download pipelines info from OmniGenBench Hub, the error is: {}".format(
                    e
                )
            )
            try:
                with open("./pipelines_info.json", "r", encoding="utf8") as f:
                    pipelines_info = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError("pipelines_info.json not found in local cache and download failed.")

    if isinstance(keyword, str):
        filtered_pipelines_info = {}
        for key in pipelines_info:
            if keyword in key:
                filtered_pipelines_info[key] = pipelines_info[key]
        return filtered_pipelines_info
    else:
        return pipelines_info


def query_benchmarks_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    This function retrieves benchmark information from the OmniGenome hub,
    either from a remote repository or from a local cache. It supports
    filtering by keywords to find specific benchmarks.

    Args:
        keyword (Union[list, str]): A keyword or list of keywords to filter benchmarks.
        repo (str, optional): The repository URL to query. If None, uses the default hub.
        local_only (bool): Whether to use only local cache. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns:
        Dict[str, Any]: A dictionary containing benchmark information filtered by the keyword.

    Example:
        >>> # Query all benchmarks
        >>> benchmarks = query_benchmarks_info("")
        >>> print(len(benchmarks))  # Number of available benchmarks
        >>> # Query specific benchmarks
        >>> benchmarks = query_benchmarks_info("RGB")
        >>> print(benchmarks.keys())  # Benchmarks containing "RGB"
    """
    if local_only:
        with open("./benchmarks_info.json", "r", encoding="utf8") as f:
            benchmarks_info = json.load(f)
    else:
        hub_repo = (repo if repo else default_omnigenbench_hub_repo) + "resolve/main/"
        try:
            response = requests.get(hub_repo + "benchmarks/benchmarks_info.json")
            benchmarks_info = response.json()
            with open("./benchmarks_info.json", "w", encoding="utf8") as f:
                json.dump(benchmarks_info, f)
        except Exception as e:
            fprint(
                "Fail to download benchmarks info from OmniGenBench Hub, the error is: {}".format(
                    e
                )
            )
            try:
                with open("./benchmarks_info.json", "r", encoding="utf8") as f:
                    benchmarks_info = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError("benchmarks_info.json not found in local cache and download failed.")

    if isinstance(keyword, str):
        filtered_benchmarks_info = {}
        for key in benchmarks_info:
            if keyword in key:
                filtered_benchmarks_info[key] = benchmarks_info[key]
        return filtered_benchmarks_info
    else:
        return benchmarks_info


def download_model(
    config_or_model: str, local_only: bool = False, repo: str = None, cache_dir=None
) -> str:
    """
    Downloads a model from a given URL. It supports both remote and local-only modes.

    Args:
        config_or_model (str): The name or path of the model to download.
        local_only (bool): A flag indicating whether to download the model from
                          the local cache. Defaults to False.
        repo (str, optional): The URL of the repository to download the model from.
        cache_dir (str, optional): The directory to cache the downloaded model.
                                 If None, uses "__OMNIGENBENCH_DATA__/models/".

    Returns:
        str: A string representing the path to the downloaded model.

    Raises:
        ConnectionError: If the model download fails.
        ValueError: If the model is not found in the repository.

    Example:
        >>> # Download a model
        >>> model_path = download_model("DNABERT-2")
        >>> print(model_path)  # Path to the downloaded model
        >>> # Download with custom cache directory
        >>> model_path = download_model("DNABERT-2", cache_dir="./models")
    """
    cache_dir = (cache_dir if cache_dir else "__OMNIGENBENCH_DATA__") + "/models/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    ckpt_config = findfile.find_files(cache_dir, ["config.json"])
    if ckpt_config:
        return os.path.dirname(ckpt_config[0])

    if local_only:
        try:
            with open("./models_info.json", "r", encoding="utf8") as f:
                models_info = json.load(f)
        except FileNotFoundError:
            fprint(
                "Local models_info.json not found. Please run the script without local_only=True to download it."
            )
            raise FileNotFoundError("models_info.json not found in local cache.")
    else:
        # Download from new Hub structure
        hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
        try:
            fprint(f"Downloading models_info.json from OmniGenBench Hub: {hub_repo}")
            response = requests.get(hub_repo + "models/models_info.json")
            response.raise_for_status()
            models_info = response.json()
            with open("./models_info.json", "w", encoding="utf8") as f:
                json.dump(models_info, f)
        except Exception as e:
            fprint(f"Failed to download models_info.json: {e}")
            # Fallback to local cache if remote download fails
            fprint("Using local cache for models_info.json if available.")
            try:
                with open("./models_info.json", "r", encoding="utf8") as f:
                    models_info = json.load(f)
            except FileNotFoundError:
                raise ConnectionError(
                    f"Failed to download models_info.json from {hub_repo} and no local cache found."
                )

    if config_or_model in models_info:
        model_info = models_info[config_or_model]
        try:
            # Download from OmniGenBench Hub
            hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
            model_url = f'{hub_repo}models/{model_info["filename"]}'
            
            fprint(f"Downloading model from OmniGenBench Hub: {model_url}")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            cache_path = os.path.join(cache_dir, f"{model_info['filename']}")
            
            # Get content length with fallback
            total_size = int(response.headers.get("content-length", 0))
            total_mb = total_size // 1024 // 1024 if total_size > 0 else None
            
            with open(cache_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    response.iter_content(chunk_size=1024 * 1024),
                    unit="MB",
                    total=total_mb,
                    desc="Downloading model",
                ):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
        except Exception as e:
            raise ConnectionError("Fail to download model: {}".format(e))

        return unzip_checkpoint(cache_path)

    else:
        raise ValueError("Model not found in the repository.")


def download_pipeline(
    pipeline_name_or_path: str,
    local_only: bool = False,
    repo: str = None,
    cache_dir=None,
) -> str:
    """
    Downloads a pipeline from a given URL. It supports both remote and local-only modes.

    Args:
        pipeline_name_or_path (str): The name or path of the pipeline to download.
        local_only (bool): A flag indicating whether to download the pipeline from
                          the local cache. Defaults to False.
        repo (str, optional): The URL of the repository to download the pipeline from.
        cache_dir (str, optional): The directory to cache the downloaded pipeline.
                                 If None, uses "__OMNIGENBENCH_DATA__/pipelines/".

    Returns:
        str: A string representing the path to the downloaded pipeline.

    Raises:
        ConnectionError: If the pipeline download fails.
        ValueError: If the pipeline is not found in the repository.

    Example:
        >>> # Download a pipeline
        >>> pipeline_path = download_pipeline("classification_pipeline")
        >>> print(pipeline_path)  # Path to the downloaded pipeline
    """
    cache_dir = (cache_dir if cache_dir else "__OMNIGENBENCH_DATA__") + "/pipelines/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    ckpt_config = findfile.find_files(cache_dir, ["config.json"])
    if ckpt_config:
        return os.path.dirname(ckpt_config[0])

    if local_only:
        with open("./pipelines_info.json", "r", encoding="utf8") as f:
            pipelines_info = json.load(f)
    else:
        # Download from OmniGenBench Hub
        hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
        try:
            fprint(f"Downloading pipelines_info.json from OmniGenBench Hub: {hub_repo}")
            response = requests.get(hub_repo + "pipelines/pipelines_info.json")
            response.raise_for_status()
            pipelines_info = response.json()
            with open("./pipelines_info.json", "w", encoding="utf8") as f:
                json.dump(pipelines_info, f)
        except Exception as e:
            fprint(f"Failed to download pipelines_info.json: {e}")
            try:
                with open("./pipelines_info.json", "r", encoding="utf8") as f:
                    pipelines_info = json.load(f)
            except FileNotFoundError:
                raise ConnectionError(
                    f"Failed to download pipelines_info.json from {hub_repo} and no local cache found."
                )

    if pipeline_name_or_path in pipelines_info:
        pipeline_info = pipelines_info[pipeline_name_or_path]
        try:
            # Download from OmniGenBench Hub
            hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
            pipeline_url = f'{hub_repo}pipelines/{pipeline_info["filename"]}'
            
            fprint(f"Downloading pipeline from OmniGenBench Hub: {pipeline_url}")
            response = requests.get(pipeline_url, stream=True)
            response.raise_for_status()
            cache_path = os.path.join(cache_dir, f"{pipeline_info['filename']}")
            
            # Get content length with fallback
            total_size = int(response.headers.get("content-length", 0))
            total_mb = total_size // 1024 // 1024 if total_size > 0 else None
            
            with open(cache_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    response.iter_content(chunk_size=1024 * 1024),
                    unit="MB",
                    total=total_mb,
                    desc="Downloading pipeline",
                ):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
        except Exception as e:
            raise ConnectionError("Fail to download pipeline: {}".format(e))

        return unzip_checkpoint(cache_path)

    else:
        raise ValueError("Pipeline not found in the repository.")


def download_benchmark(
    benchmark_name_or_path: str,
    local_only: bool = False,
    repo: str = None,
    cache_dir=None,
    use_hf_api: bool = True,
    force_download: bool = False,
) -> str:
    """
    Downloads a benchmark dataset from HuggingFace Hub or OmniGenome repository.

    **Unified Download Strategy**: This function now uses the same robust download logic
    as dataset downloads. It prioritizes HuggingFace Hub API downloads via ``snapshot_download``,
    eliminating Git-LFS dependencies and providing consistent behavior across the framework.

    Args:
        benchmark_name_or_path (str): The name or path of the benchmark to download.
            Can be a local path, a benchmark name (e.g., "RGB"), or a HuggingFace
            dataset identifier (e.g., "yangheng/OmniGenBench_RGB").
        local_only (bool): A flag indicating whether to download the benchmark from
            the local cache only. Defaults to False.
        repo (str, optional): The URL of the OmniGenome Space repository to download from.
            If None, uses the default OmniGenome hub. Only used for legacy HTTP download.
        cache_dir (str, optional): The directory to cache the downloaded benchmark.
            If None, uses "__OMNIGENBENCH_DATA__/benchmarks/".
        use_hf_api (bool): Whether to use HuggingFace Hub API for robust downloading.
            Defaults to True (recommended). Set to False to use legacy HTTP download.
        force_download (bool): Whether to re-download even if files exist in cache.
            Defaults to False.

    Returns:
        str: Path to the downloaded benchmark directory.

    Raises:
        ConnectionError: If the benchmark download fails.
        ValueError: If the benchmark is not found in any repository.
        ImportError: If huggingface_hub is not installed when use_hf_api=True.

    Example:
        >>> # Download benchmark with robust HF Hub API (recommended)
        >>> benchmark_path = download_benchmark("RGB")
        >>> print(benchmark_path)  # __OMNIGENBENCH_DATA__/benchmarks/RGB

        >>> # Download from HuggingFace Hub dataset repository
        >>> benchmark_path = download_benchmark(
        ...     "yangheng/OmniGenBench_RGB",
        ...     use_hf_api=True
        ... )

        >>> # Force re-download to update cached benchmark
        >>> benchmark_path = download_benchmark("RGB", force_download=True)

        >>> # Download with custom cache directory
        >>> benchmark_path = download_benchmark("RGB", cache_dir="./my_benchmarks")

    Note:
        **HuggingFace Hub API Method** (``use_hf_api=True``):

        - Uses ``huggingface_hub.snapshot_download()`` for direct HTTPS downloads
        - No Git or Git-LFS installation required
        - Consistent with dataset download logic
        - Automatic integrity verification and resume support
        - Requires ``huggingface_hub>=0.20.0`` package

        **Legacy HTTP Method** (``use_hf_api=False``):

        - Downloads from OmniGenome Space repository via requests
        - Fallback method when HF Hub is unavailable
        - Downloads as .zip and extracts automatically
    """
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(
            os.getcwd(),
            f"__OMNIGENBENCH_DATA__/benchmarks"
        )

    # Check if benchmark exists locally first
    # First try absolute path exists check (more reliable for paths outside cwd)
    if os.path.exists(benchmark_name_or_path):
        # Check if it has metadata.py to confirm it's a valid benchmark
        metadata_path = os.path.join(benchmark_name_or_path, "metadata.py")
        if os.path.exists(metadata_path):
            # For local paths (not hub names), always return them even with force_download
            # force_download only applies to cached hub downloads, not local paths
            fprint("Benchmark:", benchmark_name_or_path, "found locally.")
            return os.path.abspath(benchmark_name_or_path)
    
    # Also try findfile's find_cwd_dir for paths in current working directory
    p = findfile.find_dir(cache_dir, benchmark_name_or_path)
    if p:
        # Check if the found path contains metadata.py
        metadata_path = os.path.join(p, "metadata.py")
        if os.path.exists(metadata_path):
            # Local path found with metadata.py - return it
            fprint("Benchmark:", benchmark_name_or_path, "found in {}.".format(p))
            return p
        else:
            # Path exists but no metadata.py - might be a partial download, continue to download
            fprint("Path {} found but no metadata.py, will attempt download.".format(p))
    else:
        # Not a local path - must be a hub benchmark name
        fprint(
            "Benchmark:",
            benchmark_name_or_path,
            "cannot be found locally. Searching online hub to download...",
        )
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    # Check if already in cache
    bench_config = findfile.find_file(
        cache_dir, [benchmark_name_or_path, "metadata.py"]
    )
    if bench_config and not force_download:
        benchmark_path = os.path.dirname(bench_config)
        fprint(f"Benchmark found in cache: {benchmark_path}")
        return benchmark_path

    # Fallback: Legacy HTTP download from OmniGenBench Hub
    # Use the new unified Hub repository
    if local_only:
        with open("./benchmarks_info.json", "r", encoding="utf8") as f:
            benchmarks_info = json.load(f)
    else:
        # Download from OmniGenBench Hub
        hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
        try:
            # Download benchmarks_info.json from Hub
            response = requests.get(hub_repo + "benchmarks/benchmarks_info.json")
            response.raise_for_status()
            benchmarks_info = response.json()
            with open("./benchmarks_info.json", "w", encoding="utf8") as f:
                json.dump(benchmarks_info, f)
        except Exception as e:
            fprint(
                "Failed to download benchmarks_info.json from OmniGenBench Hub: {}".format(e)
            )
            try:
                with open("./benchmarks_info.json", "r", encoding="utf8") as f:
                    benchmarks_info = json.load(f)
            except FileNotFoundError:
                raise ConnectionError(
                    f"Benchmark '{benchmark_name_or_path}' not found and benchmarks_info.json is missing. "
                    f"Please check your internet connection or benchmark name."
                )

    # Extract benchmark name if it's a HF repo identifier
    benchmark_name = benchmark_name_or_path.split("/")[-1]
    if benchmark_name.startswith("OmniGenBench_"):
        benchmark_name = benchmark_name.replace("OmniGenBench_", "")

    if benchmark_name in benchmarks_info:
        benchmarks_info_item = benchmarks_info[benchmark_name]
        try:
            # Download from OmniGenBench Hub
            hub_repo = default_omnigenbench_hub_repo + "resolve/main/"
            benchmark_url = f'{hub_repo}benchmarks/{benchmarks_info_item["filename"]}'
            
            fprint(f"Downloading benchmark from OmniGenBench Hub: {benchmark_url}")
            response = requests.get(benchmark_url, stream=True)
            response.raise_for_status()  # Check for HTTP errors
            cache_path = os.path.join(cache_dir, f"{benchmarks_info_item['filename']}")
            if not os.path.exists(cache_path) or force_download:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                # Get content length with fallback
                total_size = int(response.headers.get("content-length", 0))
                total_mb = total_size // 1024 // 1024 if total_size > 0 else None
                
                with open(cache_path, "wb") as f:
                    for chunk in tqdm.tqdm(
                        response.iter_content(chunk_size=1024 * 1024),
                        unit="MB",
                        total=total_mb,
                        desc="Downloading benchmark",
                    ):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
            fprint(
                f"Benchmark {benchmark_name} downloaded successfully to: {cache_path}"
            )
            return unzip_checkpoint(cache_path)
        except ConnectionError as e:
            raise ConnectionError("Fail to download benchmark: {}".format(e))
    else:
        raise ValueError(
            f"Benchmark '{benchmark_name_or_path}' not found in any repository. "
            f"Available benchmarks: {list(benchmarks_info.keys()) if 'benchmarks_info' in locals() else 'unknown'}. "
            f"Tried HuggingFace Hub (yangheng/OmniGenBench_{benchmark_name}) and OmniGenome Space."
        )


def download_dataset(
    dataset_name_or_path: str,
    local_only: bool = False,
    repo: str = None,
    cache_dir=None,
    use_hf_api: bool = True,
    force_download: bool = False,
) -> str:
    """
    Downloads a dataset from HuggingFace Hub or OmniGenBench repository.

    **Unified Download Strategy**: This function uses the same robust download logic
    as benchmark downloads. It prioritizes HuggingFace Hub API downloads via ``snapshot_download``,
    eliminating Git-LFS dependencies and providing consistent behavior across the framework.

    Args:
        dataset_name_or_path (str): The name or path of the dataset to download.
            Can be a local path, a dataset name, or a HuggingFace dataset identifier.
        local_only (bool): Whether to use only local cache. Defaults to False.
        repo (str, optional): The URL of the OmniGenBench repository to download from.
            If None, uses the default OmniGenBench hub. Only used for legacy HTTP download.
        cache_dir (str, optional): The directory to cache the downloaded dataset.
            If None, uses "__OMNIGENBENCH_DATA__/datasets/".
        use_hf_api (bool): Whether to use HuggingFace Hub API for robust downloading.
            Defaults to True (recommended). Set to False to use legacy HTTP download.
        force_download (bool): Whether to re-download even if files exist in cache.
            Defaults to False.

    Returns:
        str: Path to the downloaded dataset directory.

    Raises:
        ConnectionError: If the dataset download fails.
        ValueError: If the dataset is not found in any repository.
        ImportError: If huggingface_hub is not installed when use_hf_api=True.

    Example:
        >>> # Download dataset with robust HF Hub API (recommended)
        >>> dataset_path = download_dataset("my_dataset")
        >>> print(dataset_path)  # __OMNIGENBENCH_DATA__/datasets/my_dataset

        >>> # Download from HuggingFace Hub dataset repository
        >>> dataset_path = download_dataset(
        ...     "yangheng/OmniGenBench_MyDataset",
        ...     use_hf_api=True
        ... )

        >>> # Force re-download to update cached dataset
        >>> dataset_path = download_dataset("my_dataset", force_download=True)

        >>> # Download with custom cache directory
        >>> dataset_path = download_dataset("my_dataset", cache_dir="./my_datasets")

    Note:
        **HuggingFace Hub API Method** (``use_hf_api=True``):

        - Uses ``huggingface_hub.snapshot_download()`` for direct HTTPS downloads
        - No Git or Git-LFS installation required
        - Consistent with benchmark download logic
        - Automatic integrity verification and resume support
        - Requires ``huggingface_hub>=0.20.0`` package

        **Legacy HTTP Method** (``use_hf_api=False``):

        - Downloads from OmniGenBench repository via requests
        - Fallback method when HF Hub is unavailable
        - Downloads as .zip and extracts automatically
    """
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = os.path.join(
            os.getcwd(),
            f"__OMNIGENBENCH_DATA__/datasets"
        )

    # Check if dataset exists locally first
    if os.path.exists(dataset_name_or_path):
        fprint(f"Dataset found locally: {dataset_name_or_path}")
        return dataset_name_or_path
    
    # Try findfile's find_cwd_dir for paths in current working directory
    p = findfile.find_dir(cache_dir, dataset_name_or_path)
    if p and not force_download:
        fprint(f"Dataset found in cache: {p}")
        return p
    else:
        # Not a local path - must be a hub dataset name
        fprint(
            "Dataset:",
            dataset_name_or_path,
            "cannot be found locally. Searching online hub to download...",
        )
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    hub_repo = (repo if repo else default_omnigenbench_hub_repo) + "resolve/main/"
    
    # Extract dataset name
    dataset_name = dataset_name_or_path.split("/")[-1]
    if dataset_name.startswith("OmniGenBench_"):
        dataset_name = dataset_name.replace("OmniGenBench_", "")
    
    # Build download URL
    url_to_download = f"{hub_repo}datasets/{dataset_name}.zip"
    local_dir = os.path.join(cache_dir, dataset_name)
    zip_path = os.path.join(cache_dir, f"{dataset_name}.zip")
    
    if not os.path.exists(local_dir) or force_download:
        try:
            fprint(f"Downloading dataset from {url_to_download}...")
            response = requests.get(url_to_download, stream=True)
            response.raise_for_status()
            
            # Get content length with fallback
            total_size = int(response.headers.get("content-length", 0))
            
            with open(zip_path, "wb") as f:
                if total_size > 0:
                    from tqdm import tqdm
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {dataset_name}",
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            fprint(f"Downloaded {zip_path}")
            
            # Unzip the dataset
            import zipfile
            fprint(f"Extracting dataset to {local_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(local_dir)
            os.remove(zip_path)
            fprint(f"Dataset extracted to: {local_dir}")
            
        except Exception as e:
            raise ConnectionError(f"Failed to download dataset: {e}")
    else:
        fprint(f"Dataset already exists at: {local_dir}")
    
    return local_dir


def check_version(repo: str = None) -> None:
    """
    Checks the version compatibility between local and remote OmniGenome.

    Args:
        repo (str, optional): The repository URL to check. If None, uses the default hub.

    Example:
        >>> check_version()  # Check version compatibility
    """
    hub_repo = (repo if repo else default_omnigenbench_hub_repo) + "resolve/main/"
    try:
        response = requests.get(hub_repo + "version.json")
        version_info = response.json()
        remote_version = version_info["version"]
        if Version(current_version) < Version(remote_version):
            fprint(
                colored(
                    f"Warning: Your local OmniGenBench version ({current_version}) "
                    f"is older than the remote version ({remote_version}). "
                    f"Please consider updating.",
                    "yellow",
                )
            )
        elif Version(current_version) > Version(remote_version):
            fprint(
                colored(
                    f"Warning: Your local OmniGenBench version ({current_version}) "
                    f"is newer than the remote version ({remote_version}). "
                    f"This might cause compatibility issues.",
                    "yellow",
                )
            )
        else:
            fprint(
                colored(
                    f"OmniGenBench version ({current_version}) is up to date.",
                    "green",
                )
            )
    except Exception as e:
        fprint(
            colored(
                f"Failed to check version: {e}",
                "red",
            )
        )
