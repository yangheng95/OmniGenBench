# -*- coding: utf-8 -*-
# file: auto_bench.py
# time: 11:54 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import os
import time
import warnings

import findfile
import torch
from metric_visualizer import MetricVisualizer

from transformers import TrainingArguments, Trainer as HFTrainer
from ...src.abc.abstract_tokenizer import OmniTokenizer
from ...src.lora.lora_model import OmniLoraModel
from ...src.misc.utils import (
    seed_everything,
    fprint,
    load_module_from_path,
    check_bench_version,
    clean_temp_checkpoint,
)
from ...src.trainer.trainer import Trainer
from ...src.trainer.accelerate_trainer import AccelerateTrainer
from ...utility.hub_utils import download_benchmark
from ... import __version__ as omnigenome_version


class AutoBench:
    """
    AutoBench is a class for automatically benchmarking genomic foundation models.

    This class provides a comprehensive framework for evaluating genomic models
    across multiple benchmarks and tasks. It handles loading benchmarks, models,
    tokenizers, and running evaluations with proper metric tracking and result
    visualization.

    AutoBench supports various evaluation scenarios including:
    - Single model evaluation across multiple benchmarks
    - Multi-seed evaluation for robustness testing
    - Different trainer backends (native, accelerate, huggingface)
    - Automatic metric visualization and result tracking

    Attributes:
        benchmark (str): The name or path of the benchmark to use.
        model_name_or_path (str): The name or path of the model to evaluate.
        tokenizer: The tokenizer to use for evaluation.
        autocast (str): The autocast precision to use ('fp16', 'bf16', etc.).
        overwrite (bool): Whether to overwrite existing evaluation results.
        trainer (str): The trainer to use ('native', 'accelerate', 'hf_trainer').
        mv_path (str): Path to the metric visualizer file.
        mv (MetricVisualizer): The metric visualizer instance.
        bench_metadata: Metadata about the benchmark configuration.
    """

    def __init__(
        self,
        benchmark,
        model_name_or_path,
        tokenizer=None,
        **kwargs,
    ):
        """
        Initializes the AutoBench instance.

        Args:
            benchmark (str): The name or path of the benchmark to use.
            model_name_or_path (str): The name or path of the model to evaluate.
            tokenizer: The tokenizer to use. If None, it will be loaded from the model path.
            **kwargs: Additional keyword arguments.
                - autocast (str): The autocast precision to use ('fp16', 'bf16', etc.).
                  Defaults to 'fp16'.
                - overwrite (bool): Whether to overwrite existing evaluation results.
                  Defaults to False.
                - trainer (str): The trainer to use ('native', 'accelerate', 'hf_trainer').
                  Defaults to 'native'.

        Example:
            >>> # Initialize with a benchmark and model
            >>> bench = AutoBench("RGB", "model_name")

            >>> # Initialize with custom settings
            >>> bench = AutoBench("RGB", "model_name",
            ...                   autocast="bf16", trainer="accelerate")
        """
        self.benchmark = benchmark.rstrip("/")
        self.autocast = kwargs.pop("autocast", "fp16")
        self.overwrite = kwargs.pop("overwrite", False)
        self.trainer = kwargs.pop("trainer", "native")

        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        if isinstance(self.model_name_or_path, str):
            self.model_name_or_path = self.model_name_or_path.rstrip("/")
            self.model_name = self.model_name_or_path.split("/")[-1]
        else:
            self.model_name = self.model_name_or_path.__class__.__name__
        if isinstance(tokenizer, str):
            self.tokenizer = tokenizer.rstrip("/")
        os.makedirs("./autobench_evaluations", exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        mv_name = f"{benchmark}-{self.model_name}"
        self.mv_path = f"./autobench_evaluations/{mv_name}-{time_str}.mv"

        mv_paths = findfile.find_files(
            "./autobench_evaluations",
            and_key=[benchmark, self.model_name, ".mv"],
        )
        if mv_paths and not self.overwrite:
            self.mv = MetricVisualizer.load(mv_paths[-1])
            self.mv.summary(round=4)
        else:
            self.mv = MetricVisualizer(self.mv_path)
        if not os.path.exists(self.benchmark):
            fprint(
                "Benchmark:",
                benchmark,
                "does not exist. Search online for available benchmarks.",
            )
            self.benchmark = download_benchmark(self.benchmark)

        # Import benchmark list
        self.bench_metadata = load_module_from_path(
            f"bench_metadata", f"{self.benchmark}/metadata.py"
        )
        check_bench_version(
            self.bench_metadata.__omnigenome_version__, omnigenome_version
        )
        fprint("Loaded benchmarks: ", self.bench_metadata.bench_list)
        self.bench_info()

    def bench_info(self):
        """
        Prints and returns information about the current benchmark setup.

        This method provides a comprehensive overview of the current
        benchmark configuration, including benchmark details, model information,
        and evaluation settings.

        Returns:
            str: A string containing benchmark information.

        Example:
            >>> info = bench.bench_info()
            >>> print(info)
        """
        info = f"Benchmark Root: {self.benchmark}\n"
        info += f"Benchmark List: {self.bench_metadata.bench_list}\n"
        info += f"Model Name or Path: {self.model_name}\n"
        info += f"Tokenizer: {self.tokenizer}\n"
        info += f"Metric Visualizer Path: {self.mv_path}\n"
        info += f"BenchConfig Details: {self.bench_metadata}\n"
        fprint(info)
        return info

    def run(self, **kwargs):
        """
        Runs the benchmarking process.

        This method iterates through the tasks in the benchmark, loads the corresponding
        configurations, initializes the model, tokenizer, and datasets, and then
        trains and evaluates the model. It supports multiple evaluation seeds and
        various trainer backends.

        Args:
            **kwargs: Additional keyword arguments that will override the default
                     parameters in the benchmark configuration.

        Example:
            >>> # Run benchmarking with default settings
            >>> bench.run()

            >>> # Run with custom parameters
            >>> bench.run(learning_rate=1e-4, batch_size=16)
        """
        bs_scale = kwargs.pop("bs_scale", 1)
        # Import benchmark config
        for _, bench in enumerate(self.bench_metadata.bench_list):
            clean_temp_checkpoint(1)  # clean temp checkpoint older than 1 day
            fprint(
                ">" * 80,
                f"\nRunning evaluation for task: {bench}",
                "Progress: ",
                _ + 1,
                "/",
                len(self.bench_metadata.bench_list),
                f"{(_ + 1) * 100 / len(self.bench_metadata.bench_list)}%",
            )
            bench_config_path = findfile.find_file(
                self.benchmark, and_key=f"{self.benchmark}.{bench}.config".split(".")
            )
            config = load_module_from_path("config", bench_config_path)
            bench_config = config.bench_config
            fprint(f"Loaded config for {bench} from {bench_config_path}")
            fprint(bench_config)

            # Init Tokenizer and Model
            if not self.tokenizer:
                tokenizer = OmniTokenizer.from_pretrained(
                    self.model_name_or_path,
                    trust_remote_code=bench_config.get("trust_remote_code", True),
                    **bench_config,
                )
            else:
                tokenizer = self.tokenizer

            if not isinstance(bench_config["seeds"], list):
                bench_config["seeds"] = [bench_config["seeds"]]

            random_seeds = bench_config["seeds"]
            for seed in random_seeds:
                _kwargs = kwargs.copy()
                for key, value in _kwargs.items():
                    if key in bench_config:
                        fprint(
                            "Override",
                            key,
                            "with",
                            value,
                            "according to the input kwargs",
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

                fprint(
                    f"AutoBench Config for {bench}:",
                    "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
                )
                for key, value in _kwargs.items():
                    if key in bench_config:
                        fprint(
                            "Override",
                            key,
                            "with",
                            value,
                            "according to the input kwargs",
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

                fprint(
                    f"AutoBench Config for {bench}:",
                    "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
                )

                batch_size = (
                    bench_config["batch_size"] if "batch_size" in bench_config else 8
                ) * bs_scale

                record_name = f"{self.benchmark}-{bench}-{self.model_name}".split("/")[
                    -1
                ]
                # check if the record exists
                if record_name in self.mv.transpose() and len(
                    list(self.mv.transpose()[record_name].values())[0]
                ) >= len(bench_config["seeds"]):
                    continue

                seed_everything(seed)
                if self.model_name_or_path:
                    model_cls = bench_config["model_cls"]
                    model = model_cls(
                        self.model_name_or_path,
                        tokenizer=tokenizer,
                        label2id=bench_config.label2id,
                        num_labels=bench_config["num_labels"],
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    raise ValueError(
                        "model_name_or_path is not specified. Please provide a valid model name or path."
                    )

                fprint(f"\n{model}")

                if kwargs.get("lora_config", None) is not None:
                    fprint(
                        "Applying LoRA to the model with config:", kwargs["lora_config"]
                    )
                    model = OmniLoraModel(model, **kwargs.get("lora_config", {}))

                # Init Trainer
                dataset_cls = bench_config["dataset_cls"]

                if hasattr(model.config, "max_position_embeddings"):
                    max_length = min(
                        bench_config["max_length"],
                        model.config.max_position_embeddings,
                    )
                else:
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

                if self.trainer == "hf_trainer":
                    # Set up HuggingFace Trainer
                    hf_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if hasattr(TrainingArguments, k) and k != "output_dir"
                    }
                    training_args = TrainingArguments(
                        output_dir=f"./autobench_evaluations/{self.model_name}-{bench}",
                        num_train_epochs=hf_kwargs.pop(
                            "num_train_epochs", bench_config["epochs"]
                        ),
                        per_device_train_batch_size=hf_kwargs.pop(
                            "batch_size", batch_size
                        ),
                        per_device_eval_batch_size=hf_kwargs.pop(
                            "batch_size", batch_size
                        ),
                        gradient_accumulation_steps=hf_kwargs.pop(
                            "gradient_accumulation_steps", 1
                        ),
                        learning_rate=hf_kwargs.pop("learning_rate", 2e-5),
                        weight_decay=hf_kwargs.pop("weight_decay", 0),
                        eval_strategy=hf_kwargs.pop("eval_strategy", "epoch"),
                        save_strategy=hf_kwargs.pop("save_strategy", "epoch"),
                        fp16=hf_kwargs.pop("fp16", True),
                        remove_unused_columns=False,
                        label_names=["labels"],
                        **hf_kwargs,
                    )

                    valid_set = valid_set if len(valid_set) else test_set

                    if len(bench_config["compute_metrics"]) > 1:
                        fprint(
                            "Multiple metrics not supported by HFTrainer, using the first one metric only."
                        )
                    trainer = HFTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_set,
                        eval_dataset=valid_set,
                        compute_metrics=(
                            bench_config["compute_metrics"][0]
                            if isinstance(bench_config["compute_metrics"], list)
                            else bench_config["compute_metrics"]
                        ),
                    )

                    # Train and evaluate
                    eval_result = trainer.evaluate(
                        valid_set if len(valid_set) else test_set
                    )
                    print(eval_result)
                    train_result = trainer.train()
                    eval_result = trainer.evaluate()
                    test_result = trainer.evaluate(
                        test_set if len(test_set) else valid_set
                    )

                    metrics = {
                        "train": train_result.metrics,
                        "eval": eval_result,
                        "test": test_result,
                    }
                    fprint(metrics)
                else:
                    optimizer = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=(
                            bench_config["learning_rate"]
                            if "learning_rate" in bench_config
                            else 2e-5
                        ),
                        weight_decay=(
                            bench_config["weight_decay"]
                            if "weight_decay" in bench_config
                            else 0
                        ),
                    )
                    if self.trainer == "accelerate":
                        trainer_cls = AccelerateTrainer
                    else:
                        trainer_cls = Trainer
                    fprint(f"Using Trainer: {trainer_cls}")
                    trainer = trainer_cls(
                        model=model,
                        train_dataset=train_set,
                        eval_dataset=valid_set,
                        test_dataset=test_set,
                        batch_size=batch_size,
                        patience=(
                            bench_config["patience"]
                            if "patience" in bench_config
                            else 3
                        ),
                        epochs=bench_config["epochs"],
                        gradient_accumulation_steps=bench_config.get(
                            "gradient_accumulation_steps", 1
                        ),
                        optimizer=optimizer,
                        loss_fn=(
                            bench_config["loss_fn"]
                            if "loss_fn" in bench_config
                            else None
                        ),
                        compute_metrics=bench_config["compute_metrics"],
                        seed=seed,
                        autocast=self.autocast,
                        **_kwargs,
                    )
                    metrics = trainer.train()

                    predictions = trainer.predictions

                    if bench_config.get("save_predictions", False):
                        os.makedirs(f"predictions/{bench}", exist_ok=True)
                        import numpy as np

                        for split in predictions.keys():
                            with open(
                                f"predictions/{bench}/{split}.npy",
                                "wb",
                            ) as f:
                                np.save(f, predictions[split])

                    if metrics:
                        for key, value in metrics["test"][-1].items():
                            try:
                                value = float(value)
                            except:
                                pass  # ignore non-float values
                            self.mv.log(f"{record_name}", f"{key}", value)
                        # for key, value in metrics['test'][-1].items():
                        #     self.mv.log(f'{record_name}', f'test_{key}', value)
                        # for i, valid_metrics in enumerate(metrics["valid"]):
                        #     for key, value in valid_metrics.items():
                        #         self.mv.log(f'{record_name}', f'valid_epoch_{i}_{key}', value)

                        self.mv.summary(round=4)
                        self.mv.dump(self.mv_path)
                        self.mv.to_csv(self.mv_path.replace(".mv", ".csv"))
                    del model, trainer, optimizer
                    torch.cuda.empty_cache()
