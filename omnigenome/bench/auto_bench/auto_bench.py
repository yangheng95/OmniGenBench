# -*- coding: utf-8 -*-
# file: auto_bench.py
# time: 11:54 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import os
import warnings

import autocuda
import findfile
import torch
from metric_visualizer import MetricVisualizer

from transformers import TrainingArguments, Trainer as HFTrainer, AutoTokenizer
from ...src.abc.abstract_tokenizer import OmniGenomeTokenizer
from ...src.misc.utils import seed_everything, fprint, load_module_from_path
from ...src.trainer.trainer import Trainer
from ...utility.hub_utils import download_benchmark


class AutoBench:
    def __init__(
        self,
        bench_root,
        model_name_or_path,
        tokenizer=None,
        use_hf_trainer=False,
        device=None,
        **kwargs,
    ):
        if not os.path.exists(bench_root):
            fprint(
                "Benchmark:",
                bench_root,
                "does not exist. Search online for available benchmarks.",
            )
            bench_root = download_benchmark(bench_root)
        self.bench_root = bench_root.rstrip("/")
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.use_hf_trainer = use_hf_trainer
        if isinstance(model_name_or_path, str):
            self.model_name_or_path = model_name_or_path.rstrip("/")
            self.model_name = self.model_name_or_path
        else:
            self.model_name = model_name_or_path.__class__.__name__
        if isinstance(tokenizer, str):
            self.tokenizer = tokenizer.rstrip("/")
        self.device = device if device else autocuda.auto_cuda()
        self.autocast = kwargs.pop("autocast", "fp16")
        self.overwrite = kwargs.pop("overwrite", False)
        # Import benchmark list
        self.bench_metadata = load_module_from_path(
            f"bench_metadata", f"{self.bench_root}/metadata.py"
        )
        fprint("Loaded benchmarks: ", self.bench_metadata.bench_list)

        self.mv_path = f"{self.bench_root}-{self.model_name}.mv".replace("/", "-")

        self.mv = MetricVisualizer(f"{self.bench_root}-{self.model_name}")
        if os.path.exists(self.mv_path) and not self.overwrite:
            self.mv = MetricVisualizer.load(self.mv_path)
            self.mv.summary(round=4)

        self.bench_info()

    def bench_info(self):
        info = f"Benchmark Root: {self.bench_root}\n"
        info += f"Benchmark List: {self.bench_metadata.bench_list}\n"
        info += f"Model Name or Path: {self.model_name}\n"
        info += f"Tokenizer: {self.tokenizer}\n"
        info += f"Device: {self.device}\n"
        info += f"Metric Visualizer Path: {self.mv_path}\n"
        info += f"BenchConfig Details: {self.bench_metadata}\n"
        fprint(info)
        return info

    def run(self, **kwargs):
        """

        :param kwargs: parameters in kwargs will be used to override the default parameters in the benchmark config
        :return:
        """

        # Import benchmark config
        for bench in self.bench_metadata.bench_list:
            _kwargs = kwargs.copy()
            bench_config_path = findfile.find_file(
                self.bench_root, f"{self.bench_root}.{bench}.config".split(".")
            )
            config = load_module_from_path("config", bench_config_path)
            bench_config = config.bench_config

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

            fprint(
                f"AutoBench Config for {bench}:",
                "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
            )

            # Init Tokenizer and Model
            if not self.tokenizer:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True
                )
            else:
                tokenizer = self.tokenizer

            if not isinstance(bench_config["seeds"], list):
                bench_config["seeds"] = [bench_config["seeds"]]

            for seed in bench_config["seeds"]:
                batch_size = (
                    bench_config["batch_size"] if "batch_size" in bench_config else 8
                )

                record_name = f"{self.bench_root}-{self.model_name}-{bench}"
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
                # Init Trainer
                dataset_cls = bench_config["dataset_cls"]

                if hasattr(model.config, "max_position_embeddings"):
                    max_length = (
                        min(
                            bench_config["max_length"],
                            model.config.max_position_embeddings,
                        )
                        - 2
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
                    shuffle=bench_config.get("shuffle", True),
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
                    shuffle=bench_config.get("shuffle", True),
                    drop_long_seq=bench_config.get("drop_long_seq", False),
                    **_kwargs,
                )

                if self.use_hf_trainer:
                    # Set up HuggingFace Trainer
                    hf_kwargs = {
                        k: v
                        for k, v in kwargs.items()
                        if hasattr(TrainingArguments, k) and k != "output_dir"
                    }
                    training_args = TrainingArguments(
                        output_dir=f"./results/{self.model_name}-{bench}",
                        num_train_epochs=hf_kwargs.get(
                            "num_train_epochs", bench_config["epochs"]
                        ),
                        per_device_train_batch_size=hf_kwargs.get(
                            "batch_size", batch_size
                        ),
                        per_device_eval_batch_size=hf_kwargs.get(
                            "batch_size", batch_size
                        ),
                        gradient_accumulation_steps=hf_kwargs.get(
                            "gradient_accumulation_steps", 1
                        ),
                        learning_rate=hf_kwargs.get("learning_rate", 2e-5),
                        weight_decay=hf_kwargs.get("weight_decay", 0),
                        eval_strategy=hf_kwargs.get("eval_strategy", "epoch"),
                        save_strategy=hf_kwargs.get("save_strategy", "epoch"),
                        fp16=hf_kwargs.get("fp16", True),
                        remove_unused_columns=False,
                        label_names=["labels"],
                        **_kwargs,
                    )

                    valid_set = valid_set if len(valid_set) else test_set

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
                        # data_collator=hf_data_collator
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
                else:
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
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
                    trainer = Trainer(
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
                        device=self.device,
                        autocast=self.autocast,
                        **_kwargs,
                    )

                    metrics = trainer.train()

                    for key, value in metrics["test"][-1].items():
                        self.mv.log(record_name, key, value)
                    self.mv.summary(round=4)
                    self.mv.dump(self.mv_path)

                    del model, trainer, optimizer
                    torch.cuda.empty_cache()

                fprint(metrics)
