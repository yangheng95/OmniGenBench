# -*- coding: utf-8 -*-
# file: trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import numpy as np
import torch
from tqdm import tqdm
import autocuda

from ..misc.utils import seed_everything, env_meta_info


class Trainer:
    def __init__(
        self,
        model,
        train_loader: torch.utils.data.DataLoader = None,
        eval_loader: torch.utils.data.DataLoader = None,
        test_loader: torch.utils.data.DataLoader = None,
        epochs: int = 3,
        optimizer: torch.optim.Optimizer = None,
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
        self.optimizer = optimizer
        self.compute_metrics = (
            compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]
        )
        self.seed = seed
        self.device = device if device else autocuda.auto_cuda()

        self.model.to(self.device)

        self.metadata = env_meta_info()
        self.metrics = {}

        self.trial_name = kwargs.get("trial_name", self.model.__class__.__name__)

    def train(self, path_to_save=None, **kwargs):
        seed_everything(self.seed)
        valid_metrics = {}
        test_metrics = {}
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = []
            train_it = tqdm(
                self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Loss:"
            )
            for batch in train_it:
                batch.to(self.device)
                loss = self.model(batch)["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_loss.append(loss.item())
                train_it.set_description(
                    f"Epoch {epoch + 1}/{self.epochs} Loss: {np.average(train_loss):.4f}"
                )

            if self.eval_loader:
                valid_metrics = self.evaluate()
                self.metrics[f"validation_metrics"] = valid_metrics
                print(f"Validation Metrics: {valid_metrics}")

            if path_to_save:
                _path_to_save = path_to_save + "_epoch_" + str(epoch)

                if valid_metrics:
                    for key, value in valid_metrics.items():
                        _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"

                self.save_model(path_to_save, **kwargs)

        if self.test_loader:
            test_metrics = self.test()
            self.metrics[f"test_metrics"] = test_metrics
            print(f"Test Metrics: {test_metrics}")

        if path_to_save:
            _path_to_save = path_to_save + "_final"

            if test_metrics:
                for key, value in test_metrics.items():
                    _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"

            self.save_model(path_to_save, **kwargs)

        return self.metrics

    def evaluate(self):
        valid_metrics = {}
        with torch.no_grad():
            self.model.eval()
            val_truth = []
            val_preds = []
            for batch in self.eval_loader:
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

    def _reload_state_dict(self, path):
        self.optimizer.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path), map_location=self.device)

    def _save_state_dict(self, path=None):
        if path is None:
            path = "init_state_dict.pt"
        self.model.to("cpu")
        torch.save(self.model.state_dict(), path)
        self.model.to(self.device)
