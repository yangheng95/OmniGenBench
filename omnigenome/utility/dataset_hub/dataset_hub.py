# -*- coding: utf-8 -*-
# File: dataset_hub.py
# Time: 02:22 20/06/2025
# Author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Website: https://yangheng95.github.io
# GitHub: https://github.com/yangheng95
# HuggingFace: https://huggingface.co/yangheng
# Google Scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All rights reserved.
import os
import warnings

import findfile
from typing_extensions import Union

from ... import OmniGenomeTokenizer, download_benchmark
from ...src.misc.utils import load_module_from_path, fprint


def load_benchmark_datasets(
    benchmark: str,
    tokenizer: Union["OmniGenomeTokenizer", str] = None,
    **kwargs: dict,
):
    if not os.path.exists(benchmark):
        fprint(
            "Benchmark:",
            benchmark,
            "does not exist. Search online for available benchmarks.",
        )
        benchmark = download_benchmark(benchmark)
        
    # Import benchmark list
    bench_metadata = load_module_from_path(
        f"bench_metadata", f"{benchmark}/metadata.py"
    )
    datasets = {}
    for _, bench in enumerate(bench_metadata.bench_list):

        bench_config_path = findfile.find_file(
            benchmark, f"{benchmark}.{bench}.config".split(".")
        )
        config = load_module_from_path("config", bench_config_path)
        bench_config = config.bench_config
        fprint(f"Loaded config for {bench} from {bench_config_path}")
        fprint(bench_config)
        _kwargs = kwargs.copy()

        # Init Tokenizer and Model
        if isinstance(tokenizer, str):
            tokenizer = OmniGenomeTokenizer.from_pretrained(
                tokenizer,
                trust_remote_code=bench_config.get("trust_remote_code", True),
                **bench_config,
            )

        for key, value in _kwargs.items():
            if key in bench_config:
                fprint(
                    "Override", key, "with", value, "according to the input kwargs"
                )
                bench_config.update({key: value})

            else:
                warnings.warn(
                    f"kwarg: {key} not found in bench_config while setting {key} = {value}"
                )
                bench_config.update({key: value})

        for key, value in bench_config.items():
            if key in bench_config and key in _kwargs:
                _kwargs.pop(key)

        if not isinstance(bench_config["seeds"], list):
            bench_config["seeds"] = [bench_config["seeds"]]

        # Init Trainer
        dataset_cls = bench_config["dataset_cls"]

        max_length = bench_config["max_length"]

        train_set = dataset_cls(
            data_source=bench_config["train_file"],
            tokenizer=tokenizer,
            label2id=bench_config["label2id"],
            max_length=max_length,
            structure_in=bench_config.get("structure_in", False),
            max_examples=bench_config.get("max_examples", None),
            shuffle=bench_config.get("shuffle", True),
            drop_long_seq=bench_config.get("drop_long_seq", False),
            **_kwargs,
        )
        test_set = dataset_cls(
            data_source=bench_config["test_file"],
            tokenizer=tokenizer,
            label2id=bench_config["label2id"],
            max_length=max_length,
            structure_in=bench_config.get("structure_in", False),
            max_examples=bench_config.get("max_examples", None),
            shuffle=False,
            drop_long_seq=bench_config.get("drop_long_seq", False),
            **_kwargs,
        )
        valid_set = dataset_cls(
            data_source=bench_config["valid_file"],
            tokenizer=tokenizer,
            label2id=bench_config["label2id"],
            max_length=max_length,
            structure_in=bench_config.get("structure_in", False),
            max_examples=bench_config.get("max_examples", None),
            shuffle=False,
            drop_long_seq=bench_config.get("drop_long_seq", False),
            **_kwargs,
        )

        dataset = {
            "train": train_set,
            "test": test_set,
            "valid": valid_set,
        }

        fprint(f"Loaded dataset for {bench} with {len(train_set)} train samples, "
              f"{len(test_set)} test samples and {len(valid_set)} valid samples.")

        datasets[bench] = dataset

    return datasets