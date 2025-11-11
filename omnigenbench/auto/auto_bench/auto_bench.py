# -*- coding: utf-8 -*-
# file: auto_bench.py
# time: 11:54 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

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
from ...src.utility.dataset_hub.dataset_hub import download_benchmark
from ...auto.config.auto_config import AutoConfig
from ... import __version__ as omnigenbench_version


class AutoBench:
    """
    Automated benchmarking framework for evaluating genomic foundation models across
    standardized benchmark suites with reproducible protocols and statistical rigor.

    This class orchestrates the complete evaluation pipeline: benchmark dataset acquisition,
    model loading, distributed inference, metric calculation, multi-seed averaging, and
    results visualization. It implements best practices for genomic machine learning evaluation,
    including proper cross-validation, ignored label handling, and task-specific metric selection.

    **Design Philosophy**: AutoBench follows the "Convention over Configuration" principle,
    providing sensible defaults while allowing full customization. By default, it uses the
    ``native`` trainer for single-GPU evaluation (optimizing for control and debuggability),
    while the CLI defaults to ``accelerate`` for distributed evaluation (optimizing for throughput).

    **Benchmark Suites Supported**:

    - **RGB**: RNA Genome Benchmarks (12 tasks) - RNA structure and function prediction
    - **BEACON**: Broad Evaluation Across Computational geNOmics (13 tasks) - Multi-domain RNA
    - **PGB**: Plant Genomics Benchmarks (7 categories) - Plant-specific sequence analysis
    - **GUE**: Genomics Understanding Evaluation (36 datasets) - DNA general understanding
    - **GB**: Genomics Benchmarks (9 datasets) - Classic DNA classification tasks

    **Evaluation Protocol**:

    1. **Dataset Loading**: Automatically downloads benchmark datasets from HuggingFace Hub
       or local cache, validates data format, and applies task-specific preprocessing
    2. **Model Initialization**: Loads pre-trained models with proper task-specific heads,
       handling multi-label classification, regression, and token-level prediction
    3. **Multi-Seed Evaluation**: Runs independent training/evaluation with different random
       seeds (typically 3-5) to quantify variance and ensure statistical significance
    4. **Metric Calculation**: Computes task-appropriate metrics (MCC, F1, AUPRC for
       classification; MSE, Spearman for regression) with proper handling of ignored labels
    5. **Result Aggregation**: Calculates mean ± standard deviation across seeds, generates
       visualizations, and serializes results with MetricVisualizer

    **Trainer Backend Selection**:

    - ``native`` (Python API default): Pure PyTorch training loop for single-GPU evaluation,
      providing explicit control over training dynamics and simplified debugging
    - ``accelerate`` (CLI default): HuggingFace Accelerate for distributed evaluation across
      multiple GPUs, enabling efficient parallel inference on large benchmarks
    - ``hf_trainer``: HuggingFace Trainer API integration for users familiar with that ecosystem

    Attributes:
        benchmark (str): Name or local path of the benchmark suite to evaluate on.
        config_or_model (str): HuggingFace Hub identifier or local path to the model.
        tokenizer: Tokenizer instance for sequence preprocessing. Auto-loaded if None.
        autocast (str): Mixed precision mode ('fp16', 'bf16', 'fp32') for memory efficiency.
        overwrite (bool): Whether to overwrite existing evaluation results or resume from cache.
        trainer (str): Training backend ('native', 'accelerate', 'hf_trainer').
        mv_path (str): Path to MetricVisualizer file for result serialization and visualization.
        mv (MetricVisualizer): Active visualizer instance for tracking metrics across seeds.
        bench_metadata: Benchmark configuration metadata loaded from benchmark's metadata.py.
    """

    def __init__(
        self,
        benchmark,
        config_or_model,
        tokenizer=None,
        **kwargs,
    ):
        """
        Initializes the AutoBench instance.

        Args:
            benchmark (str): The name or path of the benchmark to use.
                            Can be a local path or a HuggingFace Hub benchmark name.
                            For hub benchmarks, it will be automatically downloaded.
            config_or_model (str): The name or path of the model to evaluate.
            tokenizer: The tokenizer to use. If None, it will be loaded from the model path.
            **kwargs: Additional keyword arguments.
                - autocast (str): The autocast precision to use ('fp16', 'bf16', etc.).
                  Defaults to 'fp16'.
                - overwrite (bool): Whether to overwrite existing evaluation results.
                  Defaults to False.
                - trainer (str): The trainer to use ('native', 'accelerate', 'hf_trainer').
                  Defaults to 'native'.
                - cache_dir (str): Directory to cache downloaded benchmarks from hub.
                  Defaults to './__OMNIGENBENCH_DATA__/benchmarks/'.

        Example:
            >>> # Initialize with a local benchmark path
            >>> bench = AutoBench("/path/to/benchmark", "yangheng/OmniGenome-186M")

            >>> # Initialize with a HuggingFace Hub benchmark name (auto-downloads)
            >>> bench = AutoBench("RGB", "yangheng/OmniGenome-186M")

            >>> # Initialize with custom settings
            >>> bench = AutoBench("RGB", "model_name",
            ...                   autocast="bf16", trainer="accelerate")
        """
        self.benchmark_name_or_path = benchmark.rstrip("/") if isinstance(benchmark, str) else benchmark
        self.autocast = kwargs.pop("autocast", "fp16")
        self.overwrite = kwargs.pop("overwrite", False)
        self.trainer = kwargs.pop("trainer", "native")
        self.cache_dir = kwargs.pop("cache_dir", None)

        # Check if benchmark is a hub name or local path
        self.is_hub_benchmark = not os.path.exists(self.benchmark_name_or_path)
        
        if self.is_hub_benchmark:
            fprint(f"Detected HuggingFace Hub benchmark: {self.benchmark_name_or_path}")
            fprint("Downloading benchmark from hub...")
            
            # Download benchmark from hub using the unified download logic
            self.benchmark = download_benchmark(
                self.benchmark_name_or_path,
                cache_dir=self.cache_dir,
                use_hf_api=True,  # Use robust HF Hub API
                force_download=self.overwrite,
            )
            self.benchmark = os.path.dirname(findfile.find_file(self.benchmark, "metadata.py"))
            fprint(f"Benchmark downloaded to: {self.benchmark}")
        else:
            self.benchmark = self.benchmark_name_or_path
            fprint(f"Using local benchmark: {self.benchmark}")

        self.config_or_model = config_or_model
        self.tokenizer = tokenizer
        if isinstance(config_or_model, str):
            self.config_or_model = config_or_model.rstrip("/")
            self.model_name = config_or_model.split("/")[-1]
        else:
            self.model_name = config_or_model.__class__.__name__
        if isinstance(tokenizer, str):
            self.tokenizer = tokenizer.rstrip("/")
            
        os.makedirs("./autobench_evaluations", exist_ok=True)
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        
        # Use benchmark name for mv_name (not full path)
        benchmark_name = os.path.basename(self.benchmark_name_or_path)
        mv_name = f"{benchmark_name}-{self.model_name}"
        self.mv_path = f"./autobench_evaluations/{mv_name}-{time_str}.mv"

        mv_paths = findfile.find_files(
            "./autobench_evaluations",
            and_key=[benchmark_name, self.model_name, ".mv"],
        )
        if mv_paths and not self.overwrite:
            self.mv = MetricVisualizer.load(mv_paths[-1])
            self.mv.summary(round=4)
        else:
            self.mv = MetricVisualizer(self.mv_path)

        # Import benchmark list
        self.bench_metadata = load_module_from_path(
            f"bench_metadata", f"{self.benchmark}/metadata.py"
        )
        if hasattr(self.bench_metadata, "__omnigenbench_version__"):
            fprint(
                "Benchmark metadata version:",
                self.bench_metadata.__omnigenbench_version__,
            )
            check_bench_version(
                self.bench_metadata.__omnigenbench_version__, omnigenbench_version
            )
        elif hasattr(self.bench_metadata, "__omnigenome_version__"):
            fprint(
                "Benchmark metadata version:",
                self.bench_metadata.__omnigenome_version__,
            )
            check_bench_version(
                self.bench_metadata.__omnigenome_version__, omnigenbench_version
            )

        fprint("Loaded benchmarks: ", self.bench_metadata.bench_list)
        self.bench_info()

    def bench_info(self):
        """
        Prints and returns information about the current benchmark setup.

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
        Runs the benchmarking process. This method iterates through the tasks in the benchmark, loads the corresponding
        configurations, initializes the model, tokenizer, and datasets, and then
        trains and evaluates the model.

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
            _kwargs = kwargs.copy()
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
                self.benchmark,
                and_key=f"{self.benchmark}.{bench}.config".split("."),
            )
            config = load_module_from_path("config", bench_config_path)
            bench_config = None
            for attr_name in dir(config):
                attr = getattr(config, attr_name)
                if isinstance(
                    attr, AutoConfig
                ):  # Check if it is an instance of AutoConfig
                    bench_config = attr
            if bench_config is None:
                raise ValueError(
                    f"Could not find AutoConfig instance in {bench_config_path}"
                )
            fprint(f"Loaded config for {bench} from {bench_config_path}")
            fprint(bench_config)

            # Init Tokenizer and Model
            if not self.tokenizer:
                tokenizer = OmniTokenizer.from_pretrained(
                    self.config_or_model,
                    trust_remote_code=bench_config.get("trust_remote_code", True),
                    **bench_config,
                )
            else:
                tokenizer = self.tokenizer

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

            if not isinstance(bench_config["seeds"], list):
                bench_config["seeds"] = [bench_config["seeds"]]

            random_seeds = bench_config["seeds"]

            for seed in random_seeds:
                batch_size = (
                    bench_config["batch_size"] if "batch_size" in bench_config else 8
                ) * bs_scale

                record_name = f"{self.benchmark}-{bench}-{self.model_name}".split("/")[
                    -1
                ]

                # check if the record exists
                if record_name in self.mv.transpose() and len(
                    list(self.mv.transpose()[record_name].values())[0]
                ) >= len(random_seeds):
                    continue

                seed_everything(seed)
                if self.config_or_model:
                    model_cls = bench_config["model_cls"]
                    model = model_cls(
                        self.config_or_model,
                        tokenizer=tokenizer,
                        label2id=bench_config.label2id,
                        num_labels=bench_config["num_labels"],
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    raise ValueError(
                        "config_or_model is not specified. Please provide a valid model name or path."
                    )

                fprint(f"\n{model}")

                if kwargs.get("lora_config", {}) or kwargs.get("lora", True):
                    fprint(
                        "Applying LoRA to the model with config:",
                        kwargs.get("lora_config", {}) or "Default Config",
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
                    dataset_name_or_path=bench_config["train_file"],
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
                    dataset_name_or_path=bench_config["test_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=max_length,
                    structure_in=bench_config.get("structure_in", False),
                    max_examples=bench_config.get("max_examples", None),
                    shuffle=False,
                    drop_long_seq=bench_config.get("drop_long_seq", False),
                    **_kwargs,
                )
                if "valid_file" in bench_config and bench_config["valid_file"]:
                    valid_set = dataset_cls(
                        dataset_name_or_path=bench_config["valid_file"],
                        tokenizer=tokenizer,
                        label2id=bench_config["label2id"],
                        max_length=max_length,
                        structure_in=bench_config.get("structure_in", False),
                        max_examples=bench_config.get("max_examples", None),
                        shuffle=False,
                        drop_long_seq=bench_config.get("drop_long_seq", False),
                        **_kwargs,
                    )
                else:
                    valid_set = None

                if self.trainer == "hf_trainer":
                    # Set up HuggingFace Trainer
                    hf_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if hasattr(TrainingArguments, k) and k != "output_dir"
                    }
                    training_args = TrainingArguments(
                        output_dir=f"autobench_evaluations/{self.model_name}-{bench}",
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
