# -*- coding: utf-8 -*-
# file: trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch

from ..misc.utils import env_meta_info, fprint, seed_everything
from .trainer import Trainer


def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def broadcast_model(model, rank):
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)


class AccelerateTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        test_dataset: torch.utils.data.Dataset = None,
        epochs: int = 3,
        batch_size: int = 8,
        patience: int = -1,
        gradient_accumulation_steps: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        compute_metrics: [list, str] = None,
        seed: int = 42,
        autocast: str = "fp16",
        **kwargs,
    ):

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer=optimizer,
            loss_fn=loss_fn,
            compute_metrics=compute_metrics,
            seed=seed,
            autocast=autocast,
            **kwargs,
        )
        self.model = model

        self.epochs = epochs
        self.patience = patience if patience > 0 else epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compute_metrics = (
            compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]
        )
        self.seed = seed
        self._optimization_direction = None
        self.trial_name = kwargs.get("trial_name", self.model.__class__.__name__)

        # Determine mixed precision from `autocast` argument if desired
        if autocast in ["float16", "fp16"]:
            mp_setting = "fp16"
        elif autocast in ["bfloat16", "bf16"]:
            mp_setting = "bf16"
        else:
            mp_setting = "no"

        # Prepare Accelerator
        from accelerate import Accelerator
        from accelerate import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mp_setting, kwargs_handlers=[ddp_kwargs]
        )
        if self.loss_fn is not None:
            self.model.set_loss_fn(self.loss_fn)
        # 创建 dataloaders
        if kwargs.get("train_loader"):
            self.train_loader = kwargs.get("train_loader")
            self.eval_loader = kwargs.get("eval_loader", None)
            self.test_loader = kwargs.get("test_loader", None)
        else:
            self.train_loader = (
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                if train_dataset
                else None
            )
            self.eval_loader = (
                DataLoader(eval_dataset, batch_size=batch_size)
                if eval_dataset
                else None
            )
            self.test_loader = (
                DataLoader(test_dataset, batch_size=batch_size)
                if test_dataset
                else None
            )

        # 让 accelerate 处理模型和优化器的准备
        to_prepare = [self.model]
        if optimizer is not None:
            to_prepare.append(optimizer)
        if self.train_loader is not None:
            to_prepare.append(self.train_loader)
        if self.eval_loader is not None:
            to_prepare.append(self.eval_loader)
        if self.test_loader is not None:
            to_prepare.append(self.test_loader)

        prepared = self.accelerator.prepare(*to_prepare)
        self.model = prepared[0]
        idx = 1
        if optimizer is not None:
            self.optimizer = prepared[idx]
            idx += 1
        if self.train_loader is not None:
            self.train_loader = prepared[idx]
            idx += 1
        if self.eval_loader is not None:
            self.eval_loader = prepared[idx]
            idx += 1
        if self.test_loader is not None:
            self.test_loader = prepared[idx]

        self.metadata = env_meta_info()
        self.metrics = {}

    def evaluate(self):
        self.model.eval()
        all_truth = []
        all_preds = []

        # 禁用进度条在非主进程上显示
        it = tqdm(
            self.eval_loader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        )

        with torch.no_grad():
            for batch in it:
                output = self.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                # 收集所有进程的预测结果和标签
                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                # 只在主进程中处理收集到的数据
                if self.accelerator.is_main_process:
                    gathered_predictions = (
                        gathered_predictions.float().cpu().numpy(force=True)
                    )
                    gathered_labels = gathered_labels.float().cpu().numpy(force=True)
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # # 同步所有进程
        # self.accelerator.wait_for_everyone()

        # 只在主进程中计算指标
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)

            if not np.all(all_truth == -100):
                valid_metrics = {}
                for metric_func in self.compute_metrics:
                    valid_metrics.update(metric_func(all_truth, all_preds))
            else:
                valid_metrics = {
                    "Validation labels predictions may be NaN. No metrics calculated.": 0
                }

            # 打印指标信息
            fprint(valid_metrics)
        else:
            valid_metrics = None

        self.predictions.update({"valid": {"pred": all_preds, "true": all_truth}})

        return valid_metrics

    def test(self):
        self.model.eval()
        all_truth = []
        all_preds = []

        it = tqdm(
            self.test_loader,
            desc="Testing",
            disable=not self.accelerator.is_main_process,
        )

        with torch.no_grad():
            for batch in it:
                output = self.unwrap_model().predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                if self.accelerator.is_main_process:
                    gathered_predictions = (
                        gathered_predictions.float().cpu().numpy(force=True)
                    )
                    gathered_labels = gathered_labels.float().cpu().numpy(force=True)
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # # 同步所有进程
        # self.accelerator.wait_for_everyone()

        # 只在主进程中计算指标
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)

            if not np.all(all_truth == -100):
                test_metrics = {}
                for metric_func in self.compute_metrics:
                    test_metrics.update(metric_func(all_truth, all_preds))
            else:
                test_metrics = {
                    "Test labels predictions may be NaN. No metrics calculated.": 0
                }
            # 打印指标信息
            fprint(test_metrics)
        else:
            test_metrics = None

        self.predictions.update({"test": {"pred": all_preds, "true": all_truth}})

        return test_metrics

    def train(self, path_to_save=None, **kwargs):
        seed_everything(self.seed)
        # 在所有进程上创建早停标志
        early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # 确保所有进程同步启动
        self.accelerator.wait_for_everyone()

        # Initial validation or test
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            valid_metrics = self.evaluate()
        else:
            valid_metrics = self.test()

        # 在主进程中更新指标和保存模型
        if self.accelerator.is_main_process:
            if self._is_metric_better(valid_metrics, stage="valid"):
                self._save_state_dict()
                early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # 使用 all_gather 同步早停标志
        gathered_flags = self.accelerator.gather(early_stop_flag)
        early_stop_flag = (
            gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
        )  # 使用主进程的值

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            train_it = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs} Loss",
                disable=not self.accelerator.is_main_process,
            )

            # 使用 accelerator.accumulate 控制梯度累积
            for step, batch in enumerate(train_it):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                train_loss.append(loss.item())
                train_it.set_description(
                    f"Epoch {epoch + 1}/{self.epochs} Loss: {np.nanmean(train_loss):.4f}"
                )

            # 同步所有进程后再进行评估
            self.accelerator.wait_for_everyone()

            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            # 在主进程中更新指标和判断是否需要早停
            if self.accelerator.is_main_process:
                if self._is_metric_better(valid_metrics, stage="valid"):
                    self._save_state_dict()
                    early_stop_flag = torch.tensor(0, device=self.accelerator.device)
                else:
                    early_stop_flag += 1

            # 使用 all_gather 同步早停标志
            gathered_flags = self.accelerator.gather(early_stop_flag)
            early_stop_flag = (
                gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
            )  # 使用主进程的值

            # 检查是否需要早停
            if early_stop_flag.item() > self.patience:
                if self.accelerator.is_main_process:
                    fprint(f"Early stopping at epoch {epoch + 1}.")
                break

            # 只在主进程中保存检查点
            if path_to_save and self.accelerator.is_main_process:
                _path_to_save = path_to_save + "_epoch_" + str(epoch + 1)
                if valid_metrics:
                    for key, value in valid_metrics.items():
                        _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
                self.save_model(_path_to_save, **kwargs)

            # 确保所有进程同步后再进入下一轮
            self.accelerator.wait_for_everyone()

        # Final test using the best checkpoint
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            self.accelerator.wait_for_everyone()  # 确保加载完成后再测试
            test_metrics = self.test()
            if self.accelerator.is_main_process:
                self._is_metric_better(test_metrics, stage="test")

        # 只在主进程中保存最终模型
        if path_to_save and self.accelerator.is_main_process:
            _path_to_save = path_to_save + "_final"
            if self.metrics.get("test"):
                for key, value in self.metrics["test"][-1].items():
                    _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
            self.save_model(_path_to_save, **kwargs)

        self._remove_state_dict()

        self.accelerator.free_memory(
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.test_loader,
        )
        delattr(self, "accelerator")

        return self.metrics

    def predict(self, data_loader):
        return self.unwrap_model(self.model).predict(data_loader)

    def get_model(self, **kwargs):
        return self.unwrap_model(self.model)

    def compute_metrics(self):
        raise NotImplementedError(
            "The compute_metrics() function should be implemented for your model."
            " It should return a dictionary of metrics."
        )

    def save_model(self, path, overwrite=False, **kwargs):
        # Make certain only one process saves, if you're in distributed mode
        if self.accelerator.is_main_process:
            self.unwrap_model(self.model).save(path, overwrite, **kwargs)
