# -*- coding: utf-8 -*-
# file: auto_bench.py
# time: 11:54 14/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import importlib
import os
import warnings

import autocuda
import findfile
import torch
from metric_visualizer import MetricVisualizer

from ...src.abc.abstract_tokenizer import OmniGenomeTokenizer
from ...src.misc.utils import seed_everything, fprint, load_module_from_path
from ...src.trainer.trainer import Trainer


class AutoBench:
    def __init__(
        self, bench_root, model_name_or_path, tokenizer=None, device=None, **kwargs
    ):
        self.bench_root = bench_root.rstrip("/")
        self.model_name_or_path = model_name_or_path.rstrip("/")
        self.tokenizer = tokenizer.rstrip("/") if tokenizer else None
        self.device = device if device else autocuda.auto_cuda()
        self.overwrite = kwargs.pop("overwrite", False)
        # Import benchmark list
        self.bench_metadata = load_module_from_path(
            f"bench_metadata", f"{self.bench_root}/metadata.py"
        )
        fprint("Loaded benchmarks: ", self.bench_metadata.bench_list)

        self.mv_path = f"{self.bench_root}-{self.model_name_or_path}.mv".replace(
            "/", "-"
        )

        self.mv = MetricVisualizer(f"{self.bench_root}-{self.model_name_or_path}")
        if os.path.exists(self.mv_path) and not self.overwrite:
            self.mv = MetricVisualizer.load(self.mv_path)
            self.mv.summary()

        self.bench_info()

    def bench_info(self):
        info = f"Benchmark Root: {self.bench_root}\n"
        info += f"Benchmark List: {self.bench_metadata.bench_list}\n"
        info += f"Model Name or Path: {self.model_name_or_path}\n"
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
            bench_config_path = findfile.find_file(
                self.bench_root, f"{self.bench_root}.{bench}.config".split(".")
            )
            config = load_module_from_path("config", bench_config_path)
            bench_config = config.bench_config

            for key, value in kwargs.items():
                if key in bench_config:
                    fprint(
                        "Override", key, "with", value, "according to the input kwargs"
                    )
                    bench_config[key] = kwargs.get(key)
                else:
                    warnings.warn(f"Unused kwarg: {key}")

            fprint(
                f"AutoBench Config for {bench}:",
                "\n".join([f"{k}: {v}" for k, v in bench_config.items()]),
            )

            # Init Tokenizer and Model
            if self.tokenizer:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.tokenizer, trust_remote_code=True
                )
            else:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True
                )
            # Run Benchmarks
            if not isinstance(bench_config["seeds"], list):
                bench_config["seeds"] = [bench_config["seeds"]]

            for seed in bench_config["seeds"]:
                batch_size = (
                    bench_config["batch_size"] if "batch_size" in bench_config else 16
                )

                record_name = f"{self.bench_root}-{self.model_name_or_path}-{bench}"
                # check if the record exists
                if record_name in self.mv.transpose() and len(
                    list(self.mv.transpose()[record_name].values())[0]
                ) >= len(bench_config["seeds"]):
                    continue

                seed_everything(seed)
                model_cls = bench_config["model_cls"]
                model = model_cls(
                    self.model_name_or_path,
                    tokenizer=tokenizer,
                    label2id=bench_config.label2id,
                    num_labels=bench_config["num_labels"],
                    trust_remote_code=True,
                )

                # Init Trainer
                dataset_cls = bench_config["dataset_cls"]

                train_set = dataset_cls(
                    data_source=bench_config["train_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=bench_config["max_length"],
                )
                test_set = dataset_cls(
                    data_source=bench_config["test_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=bench_config["max_length"],
                )
                valid_set = dataset_cls(
                    data_source=bench_config["valid_file"],
                    tokenizer=tokenizer,
                    label2id=bench_config["label2id"],
                    max_length=bench_config["max_length"],
                )

                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True
                )
                valid_loader = torch.utils.data.DataLoader(
                    valid_set, batch_size=batch_size
                )
                test_loader = torch.utils.data.DataLoader(
                    test_set, batch_size=batch_size
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=bench_config["learning_rate"]
                    if "learning_rate" in bench_config
                    else 2e-5,
                    weight_decay=bench_config["weight_decay"]
                    if "weight_decay" in bench_config
                    else 0,
                )
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    eval_loader=valid_loader,
                    test_loader=test_loader,
                    batch_size=batch_size,
                    epochs=bench_config["epochs"],
                    patience=bench_config["patience"]
                    if "patience" in bench_config
                    else 3,
                    optimizer=optimizer,
                    loss_fn=bench_config["loss_fn"]
                    if "loss_fn" in bench_config
                    else None,
                    compute_metrics=bench_config["compute_metrics"],
                    seed=seed,
                    device=self.device,
                )

                metrics = trainer.train(
                    autocast=bench_config["autocast"]
                    if "autocast" in bench_config
                    else False
                )

                for key, value in metrics["test"][-1].items():
                    self.mv.log(record_name, key, value)
                fprint(metrics)
                self.mv.summary(round=4)
                self.mv.dump(self.mv_path)
                del model, trainer, optimizer, train_loader, valid_loader, test_loader
                torch.cuda.empty_cache()
