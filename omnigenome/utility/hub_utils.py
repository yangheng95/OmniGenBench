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
from distutils.version import StrictVersion
from typing import Union, Dict, Any

import findfile
import requests
import tqdm
from termcolor import colored

from omnigenome import __version__ as current_version
from omnigenome.src.misc.utils import fprint

default_repo = "https://huggingface.co/spaces/anonymous8/gfm_hub/"


def unzip_checkpoint(checkpoint_path):
    """
    Unzips a checkpoint file.

    :param checkpoint_path: The path to the checkpoint file.
    """
    import zipfile

    with zipfile.ZipFile(checkpoint_path, "r") as zip_ref:
        zip_ref.extractall(checkpoint_path.strip(".zip"))

    return checkpoint_path.strip(".zip")


def query_models_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
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
    if local_only:
        with open("./pipelines_info.json", "r", encoding="utf8") as f:
            pipelines_info = json.load(f)
    else:
        repo = repo if repo else default_repo
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


def query_datasets_info(
    keyword: Union[list, str], repo: str = None, local_only: bool = False, **kwargs
) -> Dict[str, Any]:
    if local_only:
        with open("./datasets_info.json", "r", encoding="utf8") as f:
            datasets_info = json.load(f)
    else:
        repo = repo if repo else default_repo
        try:
            response = requests.get(repo + "datasets_info.json")
            datasets_info = response.json()
            with open("./datasets_info.json", "w", encoding="utf8") as f:
                json.dump(datasets_info, f)
        except Exception as e:
            fprint(
                "Fail to download datasets info from huggingface space, the error is: {}".format(
                    e
                )
            )
            with open("./datasets_info.json", "r", encoding="utf8") as f:
                datasets_info = json.load(f)

    if isinstance(keyword, str):
        filtered_datasets_info = {}
        for key in datasets_info:
            if keyword in key:
                filtered_datasets_info[key] = datasets_info[key]
        return filtered_datasets_info
    else:
        return datasets_info


def download_model(
    model_name_or_path: str, local_only: bool = False, repo: str = None, cache_dir=None
) -> str:
    """
    Downloads a model from a given URL.

    :param model_name_or_path: The name or path of the model to download.
    :param local_only: A flag indicating whether to download the model from the local cache.
    :param repo: The URL of the repository to download the model from.
    :param cache_dir: The directory to cache the downloaded model.
    :return: A string representing the path to the downloaded model.
    """
    cache_dir = cache_dir if cache_dir else "__OMNIGENOME_DATA__"
    if local_only:
        with open("./models_info.json", "r", encoding="utf8") as f:
            models_info = json.load(f)
    else:
        repo = repo if repo else default_repo
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

    model_info = models_info[model_name_or_path]

    if model_name_or_path in models_info:
        ckpt_config = findfile.find_files(cache_dir, ["config.json"])
        if ckpt_config:
            return os.path.dirname(ckpt_config[0])
        try:
            model_url = f'{repo}/models/{model_info["filename"]}'
            response = requests.get(model_url, stream=True)
            cache_path = os.path.join(cache_dir, f"models/{model_info['filename']}")
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

    :param pipeline_name_or_path: The name or path of the pipeline to download.
    :param local_only: A flag indicating whether to download the pipeline from the local cache.
    :param repo: The URL of the repository to download the pipeline from.
    :param cache_dir: The directory to cache the downloaded pipeline.
    :return: A string representing the path to the downloaded pipeline.
    """
    cache_dir = cache_dir if cache_dir else "__OMNIGENOME_DATA__"
    if local_only:
        with open("./pipelines_info.json", "r", encoding="utf8") as f:
            pipelines_info = json.load(f)
    else:
        repo = repo if repo else default_repo
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

    pipeline_info = pipelines_info[pipeline_name_or_path]

    if pipeline_name_or_path in pipelines_info:
        ckpt_config = findfile.find_files(cache_dir, ["config.json"])
        if ckpt_config:
            return os.path.dirname(ckpt_config[0])
        try:
            pipeline_url = f'{repo}/pipelines/{pipeline_info["filename"]}'
            response = requests.get(pipeline_url, stream=True)
            cache_path = os.path.join(
                cache_dir, f"pipelines/{pipeline_info['filename']}"
            )
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


def check_version(repo: str = None) -> None:
    """
    Checks the version of the package.

    :param repo: The URL of the repository to check the version from.
    """
    repo = repo if repo else default_repo
    try:
        response = requests.get(repo + "version.json")
        version_info = response.json()
        latest_version = version_info["version"]
        if StrictVersion(current_version) < StrictVersion(latest_version):
            fprint(
                colored(
                    f"An updated version of the package is available. Please upgrade to version {latest_version}.",
                    "red",
                )
            )
        else:
            fprint(colored("The package is up-to-date.", "green"))
    except Exception as e:
        fprint("Fail to check the version of the package: {}".format(e))
        fprint(colored("The package is up-to-date.", "green"))
