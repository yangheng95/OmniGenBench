# -*- coding: utf-8 -*-
# file: trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import os

import autocuda
import numpy as np
import torch
from tqdm import tqdm

from ..misc.utils import env_meta_info, fprint, seed_everything

import sklearn.metrics
from hashlib import sha256


def _infer_optimization_direction(metrics, prev_metrics):
    larger_is_better_metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "pr_auc",
        "score",
        # ...
    ]
    smaller_is_better_metrics = [
        "loss",
        "error",
        "mse",
        "mae",
        "r2",
        "distance",
        # ...
    ]
    for metric in larger_is_better_metrics:
        if metric in list(prev_metrics[0].keys())[0]:
            return "larger_is_better"
    for metric in smaller_is_better_metrics:
        if metric in list(prev_metrics[0].keys())[0]:
            return "smaller_is_better"

    fprint(
        "Cannot determine the optimization direction. Trying to infer from the metrics."
    )
    is_prev_increasing = np.mean(list(prev_metrics[0].values())[0]) < np.mean(
        list(prev_metrics[-1].values())[0]
    )
    is_still_increasing = np.mean(list(prev_metrics[1].values())[0]) < np.mean(
        list(metrics.values())[0]
    )

    if is_prev_increasing and is_still_increasing:
        return "larger_is_better"

    is_prev_decreasing = np.mean(list(prev_metrics[0].values())[0]) > np.mean(
        list(prev_metrics[-1].values())[0]
    )
    is_still_decreasing = np.mean(list(prev_metrics[1].values())[0]) > np.mean(
        list(metrics.values())
    )

    if is_prev_decreasing and is_still_decreasing:
        return "smaller_is_better"


class Trainer:
    def __init__(
        self,
        model,
        train_loader: torch.utils.data.DataLoader = None,
        eval_loader: torch.utils.data.DataLoader = None,
        test_loader: torch.utils.data.DataLoader = None,
        epochs: int = 3,
        patience: int = 3,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        compute_metrics: [list, str] = None,
        seed: int = 42,
        device: [torch.device, str] = None,
        *args,
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.patience = patience
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compute_metrics = (
            compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]
        )
        self.seed = seed
        self.device = device if device else autocuda.auto_cuda()
        if self.loss_fn is not None:
            self.model.set_loss_fn(self.loss_fn)
        self.model.to(self.device)

        self.metadata = env_meta_info()
        self.metrics = {}

        self._optimization_direction = None
        self.trial_name = kwargs.get("trial_name", self.model.__class__.__name__)

    def _is_metric_better(self, metrics, stage="valid"):
        assert stage in [
            "valid",
            "test",
        ], "The metrics stage should be either 'valid' or 'test'."

        fprint(metrics)

        prev_metrics = self.metrics.get(stage, None)

        if not prev_metrics or len(prev_metrics) <= 1:
            if stage not in self.metrics:
                self.metrics.update({f"{stage}": [metrics]})
            else:
                self.metrics[f"{stage}"].append(metrics)

        if "best_valid" not in self.metrics:
            self.metrics.update({"best_valid": metrics})
            return True
        self._optimization_direction = (
            _infer_optimization_direction(metrics, prev_metrics)
            if self._optimization_direction is None
            else self._optimization_direction
        )

        if self._optimization_direction == "larger_is_better":
            if np.mean(list(metrics.values())[0]) > np.mean(
                list(self.metrics["best_valid"].values())[0]
            ):
                self.metrics.update({"best_valid": metrics})
                return True
        elif self._optimization_direction == "smaller_is_better":
            if np.mean(list(metrics.values())[0]) < np.mean(
                list(self.metrics["best_valid"].values())[0]
            ):
                self.metrics.update({"best_valid": metrics})
                return True

        return False

    def train(self, path_to_save=None, autocast=False, **kwargs):
        seed_everything(self.seed)
        patience = 0

        if self.eval_loader is not None and len(self.eval_loader) > 0:
            valid_metrics = self.evaluate()
        else:
            valid_metrics = self.test()
        if self._is_metric_better(valid_metrics, stage="valid"):
            self._save_state_dict()
            patience = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            train_it = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Loss:"
            )
            for batch in train_it:
                batch.to(self.device)
                if autocast:
                    with torch.cuda.amp.autocast():
                        loss = self.model(batch)["loss"]
                else:
                    loss = self.model(batch)["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss.append(loss.item())
                train_it.set_description(
                    f"Epoch {epoch + 1}/{self.epochs} Loss: {np.average(train_loss):.4f}"
                )

            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            if self._is_metric_better(valid_metrics, stage="valid"):
                self._save_state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    fprint(f"Early stopping at epoch {epoch + 1}.")
                    break

            if path_to_save:
                _path_to_save = path_to_save + "_epoch_" + str(epoch + 1)

                if valid_metrics:
                    for key, value in valid_metrics.items():
                        _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"

                self.save_model(_path_to_save, **kwargs)

        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            test_metrics = self.test()
            self._is_metric_better(test_metrics, stage="test")

        if path_to_save:
            _path_to_save = path_to_save + "_final"
            if self.metrics["test"]:
                for key, value in self.metrics["test"][-1].items():
                    _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"

            self.save_model(_path_to_save, **kwargs)

        self._remove_state_dict()

        return self.metrics

    def evaluate(self):
        valid_metrics = {}
        with torch.no_grad():
            self.model.eval()
            val_truth = []
            val_preds = []
            it = tqdm(self.eval_loader, desc="Evaluating")
            for batch in it:
                batch.to(self.device)
                predictions = self.model.predict(batch)["predictions"]
                val_truth.append(batch["labels"].detach().cpu().numpy())
                val_preds.append(np.array(predictions))

            val_truth = np.concatenate(val_truth)
            val_preds = np.concatenate(val_preds)
            for metric_func in self.compute_metrics:
                valid_metrics.update(metric_func(val_truth, val_preds))
            return valid_metrics

    def test(self):
        test_metrics = {}
        with torch.no_grad():
            self.model.eval()
            preds = []
            truth = []
            it = tqdm(self.test_loader, desc="Testing")
            for batch in it:
                batch.to(self.device)
                predictions = self.model.predict(batch)["predictions"]
                truth.append(batch["labels"].detach().cpu().numpy())
                preds.append(predictions)
            preds = np.concatenate(preds)
            truth = np.concatenate(truth)
            for metric_func in self.compute_metrics:
                test_metrics.update(metric_func(truth, preds))
            return test_metrics

    def predict(self, data_loader):
        return self.model.predict(data_loader)

    def get_model(self, **kwargs):
        return self.model

    def compute_metrics(self):
        raise NotImplementedError(
            "The compute_metrics() function should be implemented for your model."
            " It should return a dictionary of metrics."
        )

    def save_model(self, path, overwrite=False, **kwargs):
        self.model.save(path, overwrite, **kwargs)

    def _load_state_dict(self):
        if os.path.exists(self._model_state_dict_path):
            self.model.load_state_dict(torch.load(self._model_state_dict_path))
        self.model.to(self.device)

    def _save_state_dict(self):
        if not hasattr(self, "_model_state_dict_path"):
            self._model_state_dict_path = (
                sha256(self.__repr__().encode()).hexdigest() + "_model_state_dict.pt"
            )

        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)

        self.model.to("cpu")
        torch.save(self.model.state_dict(), self._model_state_dict_path)
        self.model.to(self.device)

    def _remove_state_dict(self):
        if not hasattr(self, "_model_state_dict_path"):
            self._model_state_dict_path = (
                sha256(self.__repr__().encode()).hexdigest() + "_model_state_dict.pt"
            )

        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)
