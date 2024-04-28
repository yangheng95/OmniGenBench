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

import autocuda
import findfile
import torch
from metric_visualizer import MetricVisualizer

from ...src.abc.abstract_tokenizer import OmniGenomeTokenizer
from ...src.trainer.trainer import Trainer
from ...src.misc.utils import seed_everything, fprint


class AutoBenchConfig:
    def __init__(self, root, model_name_or_path, tokenizer=None):
        pass

class AutoBench:
    def __init__(self, bench_root, model_name_or_path, tokenizer=None, device=None, **kwargs):
        self.bench_root = bench_root
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.device = device if device else autocuda.auto_cuda()

        # Import benchmark list
        self.bench_metadata = importlib.import_module(f"{self.bench_root}.metadata")

        self.mv_path = f"{self.bench_root}-{self.model_name_or_path}.mv".replace('/', '-')
        if os.path.exists(self.mv_path):
            self.mv = MetricVisualizer.load(self.mv_path)
            self.mv.summary()
        else:
            self.mv = MetricVisualizer(f"{self.bench_root}-{self.model_name_or_path}")

    def run(self, **kwargs):
        """

        :param kwargs: parameters in kwargs will be used to overwrite the default parameters in the benchmark config
        :return:
        """

        # Import benchmark config
        for bench in self.bench_metadata.bench_list:

            bench_config_path = findfile.find_file(
                self.bench_root, f"{self.bench_root}.{bench}.config".split(".")
            )
            config = importlib.import_module(
                bench_config_path.replace(os.sep, ".").replace(".py", "")
            )
            bench_config = config.bench_config

            for key, value in kwargs.items():
                if key in bench_config:
                    bench_config[key] = value

            # Init Tokenizer and Model
            if self.tokenizer:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.tokenizer, trust_remote_code=True
                )
            else:
                tokenizer = OmniGenomeTokenizer.from_pretrained(
                    self.model_name_or_path, trust_remote_code=True
                )

            record_name = f"{self.bench_root}-{self.model_name_or_path}-{bench}"
            # check if the record exists
            if record_name in self.mv.transpose() and len(list(self.mv.transpose()[record_name].values())[0]) >= len(bench_config["seeds"]):
                continue

            # Run Benchmarks
            for seed in bench_config["seeds"]:

                seed_everything(seed)
                model_cls = bench_config["model_cls"]
                model = model_cls(
                    self.model_name_or_path,
                    tokenizer=tokenizer,
                    label2id=bench_config.label2id,
                    num_labels=bench_config["num_labels"],
                    trust_remote_code=True,
                )
                model.to(autocuda.auto_cuda())

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
                batch_size = bench_config["batch_size"] if "batch_size" in bench_config else 8

                train_loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True
                )
                valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
                test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=bench_config["learning_rate"],
                    weight_decay=bench_config["weight_decay"],
                )
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    eval_loader=valid_loader,
                    test_loader=test_loader,
                    batch_size=batch_size,
                    epochs=bench_config["epochs"],
                    optimizer=optimizer,
                    loss_fn=bench_config["loss_fn"] if "loss_fn" in bench_config else None,
                    compute_metrics=bench_config["compute_metrics"],
                    seed=seed,
                    device=self.device,
                )
                metrics = trainer.train()
                for key, value in metrics["test"][-1].items():
                    self.mv.log(record_name, key, value)
                fprint(metrics)
                self.mv.summary()
                self.mv.dump(self.mv_path)
                del model, trainer, optimizer, train_loader, valid_loader, test_loader
                torch.cuda.empty_cache()
