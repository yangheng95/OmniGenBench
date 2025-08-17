# -*- coding: utf-8 -*-
# file: hub_utils.py
# time: 16:54 13/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import json
import os
from typing import Union, Dict, Any

import findfile
import requests
import tqdm
from packaging.version import Version
from termcolor import colored

from .. import __version__ as current_version
from ..src.misc.utils import fprint, default_omnigenome_repo


def unzip_checkpoint(checkpoint_path):
    """
    Unzips a checkpoint file.

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

    fprint("Unzipping checkpoint from {}...".format(checkpoint_path))
    with zipfile.ZipFile(checkpoint_path, "r") as zip_ref:
        zip_ref.extractall(checkpoint_path.strip(".zip"))

    return checkpoint_path.strip(".zip")


def query_models_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Queries information about available models from the hub.

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
        repo = repo if repo else "https://huggingface.co/spaces/anonymous8/gfm_hub/"
        try:
            response = requests.get(repo + "models_info.json")
            models_info = response.json()
            with open("./models_info.json", "w", encoding="utf8") as f:
                json.dump(models_info, f)
        except Exception as e:
            fprint(
                "Fail to download models info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./models_info.json", "r", encoding="utf8") as f:
                models_info = json.load(f)

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
    Queries information about available pipelines from the hub.

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
        repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
        try:
            response = requests.get(repo + "pipelines_info.json")
            pipelines_info = response.json()
            with open("./pipelines_info.json", "w", encoding="utf8") as f:
                json.dump(pipelines_info, f)
        except Exception as e:
            fprint(
                "Fail to download pipelines info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./pipelines_info.json", "r", encoding="utf8") as f:
                pipelines_info = json.load(f)

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
    Queries information about available benchmarks from the hub.

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
        repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
        try:
            response = requests.get(repo + "benchmarks_info.json")
            benchmarks_info = response.json()
            with open("./benchmarks_info.json", "w", encoding="utf8") as f:
                json.dump(benchmarks_info, f)
        except Exception as e:
            fprint(
                "Fail to download datasets info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./benchmarks_info.json", "r", encoding="utf8") as f:
                benchmarks_info = json.load(f)

    if isinstance(keyword, str):
        filtered_benchmarks_info = {}
        for key in benchmarks_info:
            if keyword in key:
                filtered_benchmarks_info[key] = benchmarks_info[key]
        return filtered_benchmarks_info
    else:
        return benchmarks_info


def download_model(
    model_name_or_path: str, local_only: bool = False, repo: str = None, cache_dir=None
) -> str:
    """
    Downloads a model from a given URL.

    This function downloads a model from the OmniGenome hub and caches it
    locally for future use. It supports both remote and local-only modes.

    Args:
        model_name_or_path (str): The name or path of the model to download.
        local_only (bool): A flag indicating whether to download the model from
                          the local cache. Defaults to False.
        repo (str, optional): The URL of the repository to download the model from.
        cache_dir (str, optional): The directory to cache the downloaded model.
                                 If None, uses "__OMNIGENOME_DATA__/models/".

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
    cache_dir = (cache_dir if cache_dir else "__OMNIGENOME_DATA__") + "/models/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
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
        repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
        try:
            response = requests.get(repo + "models_info.json")
            models_info = response.json()
            with open("./models_info.json", "w", encoding="utf8") as f:
                json.dump(models_info, f)
        except Exception as e:
            fprint(
                "Fail to download models info from huggingface space, the error is: {}".format(
                    e
                )
            )
            # Fallback to local cache if remote download fails
            fprint("Using local cache for models_info.json. Ensure it is up-to-date.")
            try:
                with open("./models_info.json", "r", encoding="utf8") as f:
                    models_info = json.load(f)
            except FileNotFoundError:
                fprint(
                    "Local models_info.json not found. Please run the script without local_only=True to download it."
                )
                raise FileNotFoundError("models_info.json not found in local cache.")

    if model_name_or_path in models_info:
        model_info = models_info[model_name_or_path]
        try:
            model_url = f'{repo}/models/{model_info["filename"]}'
            response = requests.get(model_url, stream=True)
            cache_path = os.path.join(cache_dir, f"{model_info['filename']}")
            with open(cache_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    response.iter_content(chunk_size=1024 * 1024),
                    unit="MB",
                    total=int(response.headers["content-length"]) // 1024 // 1024,
                    desc="Downloading model",
                ):
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
    Downloads a pipeline from a given URL.

    This function downloads a pipeline from the OmniGenome hub and caches it
    locally for future use. It supports both remote and local-only modes.

    Args:
        pipeline_name_or_path (str): The name or path of the pipeline to download.
        local_only (bool): A flag indicating whether to download the pipeline from
                          the local cache. Defaults to False.
        repo (str, optional): The URL of the repository to download the pipeline from.
        cache_dir (str, optional): The directory to cache the downloaded pipeline.
                                 If None, uses "__OMNIGENOME_DATA__/pipelines/".

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
    cache_dir = (cache_dir if cache_dir else "__OMNIGENOME_DATA__") + "/pipelines/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    ckpt_config = findfile.find_files(cache_dir, ["config.json"])
    if ckpt_config:
        return os.path.dirname(ckpt_config[0])

    if local_only:
        with open("./pipelines_info.json", "r", encoding="utf8") as f:
            pipelines_info = json.load(f)
    else:
        repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
        try:
            response = requests.get(repo + "pipelines_info.json")
            pipelines_info = response.json()
            with open("./pipelines_info.json", "w", encoding="utf8") as f:
                json.dump(pipelines_info, f)
        except Exception as e:
            fprint(
                "Fail to download pipelines info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./pipelines_info.json", "r", encoding="utf8") as f:
                pipelines_info = json.load(f)

    if pipeline_name_or_path in pipelines_info:
        pipeline_info = pipelines_info[pipeline_name_or_path]
        try:
            pipeline_url = f'{repo}/pipelines/{pipeline_info["filename"]}'
            response = requests.get(pipeline_url, stream=True)
            cache_path = os.path.join(cache_dir, f"{pipeline_info['filename']}")
            with open(cache_path, "wb") as f:
                for chunk in tqdm.tqdm(
                    response.iter_content(chunk_size=1024 * 1024),
                    unit="MB",
                    total=int(response.headers["content-length"]) // 1024 // 1024,
                    desc="Downloading pipeline",
                ):
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
) -> str:
    """
    Downloads a benchmark from a given URL.

    This function downloads a benchmark from the OmniGenome hub and caches it
    locally for future use. It supports both remote and local-only modes.

    Args:
        benchmark_name_or_path (str): The name or path of the benchmark to download.
        local_only (bool): A flag indicating whether to download the benchmark from
                          the local cache. Defaults to False.
        repo (str, optional): The URL of the repository to download the benchmark from.
        cache_dir (str, optional): The directory to cache the downloaded benchmark.
                                 If None, uses "__OMNIGENOME_DATA__/benchmarks/".

    Returns:
        str: A string representing the path to the downloaded benchmark.

    Raises:
        ConnectionError: If the benchmark download fails.
        ValueError: If the benchmark is not found in the repository.

    Example:
        >>> # Download a benchmark
        >>> benchmark_path = download_benchmark("RGB")
        >>> print(benchmark_path)  # Path to the downloaded benchmark

        >>> # Download with custom cache directory
        >>> benchmark_path = download_benchmark("RGB", cache_dir="./benchmarks")
    """
    p = findfile.find_cwd_dir(benchmark_name_or_path)
    if p:
        fprint("Benchmark:", benchmark_name_or_path, "found in {}.".format(p))
        return p
    else:
        fprint(
            "Benchmark:",
            benchmark_name_or_path,
            "cannot be found. Search from the online hub to download...",
        )
    cache_dir = (cache_dir if cache_dir else "__OMNIGENOME_DATA__") + "/benchmarks/"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    bench_config = findfile.find_file(
        cache_dir, [benchmark_name_or_path, "metadata.py"]
    )
    if bench_config:
        return os.path.dirname(bench_config)

    if local_only:
        with open("./benchmarks_info.json", "r", encoding="utf8") as f:
            benchmarks_info = json.load(f)
    else:
        repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
        try:
            response = requests.get(repo + "benchmarks_info.json")
            benchmarks_info = response.json()
            with open("./benchmarks_info.json", "w", encoding="utf8") as f:
                json.dump(benchmarks_info, f)
        except Exception as e:
            fprint(
                "Fail to download datasets info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./benchmarks_info.json", "r", encoding="utf8") as f:
                benchmarks_info = json.load(f)

    if benchmark_name_or_path in benchmarks_info:
        benchmarks_info_item = benchmarks_info[benchmark_name_or_path]
        try:
            benchmark_url = f'{repo}benchmarks/{benchmarks_info_item["filename"]}'
            response = requests.get(benchmark_url, stream=True)
            cache_path = os.path.join(cache_dir, f"{benchmarks_info_item['filename']}")
            if not os.path.exists(cache_path):
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as f:
                    for chunk in tqdm.tqdm(
                        response.iter_content(chunk_size=1024 * 1024),
                        unit="MB",
                        total=int(response.headers["content-length"]) // 1024 // 1024,
                        desc="Downloading benchmark",
                    ):
                        f.write(chunk)
            fprint(
                f"Benchmark {benchmark_name_or_path} downloaded successfully to: {cache_path}"
            )
            return unzip_checkpoint(cache_path)
        except ConnectionError as e:
            raise ConnectionError("Fail to download benchmark: {}".format(e))

    else:
        raise ValueError("Benchmark not found in the repository.")


def check_version(repo: str = None) -> None:
    """
    Checks the version compatibility between local and remote OmniGenome.

    This function compares the local OmniGenome version with the version
    available in the remote repository to ensure compatibility.

    Args:
        repo (str, optional): The repository URL to check. If None, uses the default hub.

    Example:
        >>> check_version()  # Check version compatibility
    """
    repo = (repo if repo else default_omnigenome_repo) + "resolve/main/"
    try:
        response = requests.get(repo + "version.json")
        version_info = response.json()
        remote_version = version_info["version"]
        if Version(current_version) < Version(remote_version):
            fprint(
                colored(
                    f"Warning: Your local OmniGenome version ({current_version}) "
                    f"is older than the remote version ({remote_version}). "
                    f"Please consider updating.",
                    "yellow",
                )
            )
        elif Version(current_version) > Version(remote_version):
            fprint(
                colored(
                    f"Warning: Your local OmniGenome version ({current_version}) "
                    f"is newer than the remote version ({remote_version}). "
                    f"This might cause compatibility issues.",
                    "yellow",
                )
            )
        else:
            fprint(
                colored(
                    f"OmniGenome version ({current_version}) is up to date.",
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
