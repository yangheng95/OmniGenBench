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

def _infer_optimization_direction(metrics, prev_metrics):
    """
    This function infers whether the optimization direction is 'larger_is_better' or 'smaller_is_better'
    based on the comparison of the current and previous metrics.
    """
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

    fprint("Cannot determine the optimisation direction. Attempting inference from the metrics.")
    is_prev_increasing = np.mean(list(prev_metrics[0].values())[0]) < np.mean(list(prev_metrics[-1].values())[0])
    is_still_increasing = np.mean(list(prev_metrics[1].values())[0]) < np.mean(list(metrics.values())[0])
    fprint(
        "Cannot determine the optimisation direction. Attempting inference from the metrics."
    )

    if is_prev_increasing and is_still_increasing:
        return "larger_is_better"

    is_prev_decreasing = np.mean(list(prev_metrics[0].values())[0]) > np.mean(list(prev_metrics[-1].values())[0])
    is_still_decreasing = np.mean(list(prev_metrics[1].values())[0]) > np.mean(list(metrics.values()))

    if is_prev_decreasing and is_still_decreasing:
        return "smaller_is_better"


class AccelerateTrainer:
    def __init__(
        self,
        model,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        test_dataset: torch.utils.data.Dataset = None,
        epochs: int = 3,
        batch_size: int = 8,
        patience: int = 3,
        gradient_accumulation_steps: int = 1,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: torch.nn.Module = None,
        compute_metrics: [list, str] = None,
        seed: int = 42,
        autocast: str = "float16",
        **kwargs,
    ):

        self.model = model

        # # Set up DataLoaders based on the given datasets or keyword arguments
        if kwargs.get("train_loader"):
            self.train_loader = kwargs.get("train_loader")
            self.eval_loader = kwargs.get("eval_loader", None)
            self.test_loader = kwargs.get("test_loader", None)
        else:
            # If no DataLoader is provided, create DataLoader from datasets
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True) if train_dataset else None
            self.eval_loader = DataLoader(eval_dataset, batch_size=batch_size) if eval_dataset else None
            self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
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

        self.epochs = epochs
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.compute_metrics = compute_metrics if isinstance(compute_metrics, list) else [compute_metrics]
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

        self.accelerator = Accelerator(mixed_precision=mp_setting, kwargs_handlers=[ddp_kwargs])

        self.accelerator = Accelerator(
            mixed_precision=mp_setting, kwargs_handlers=[ddp_kwargs]
        )
        # If loss_fn is provided, set it to the model
        if self.loss_fn is not None:
            self.model.set_loss_fn(self.loss_fn)
            # Prepare the model, optimizer, and data loaders for distributed training
            if kwargs.get("train_loader"):
                self.train_loader = kwargs.get("train_loader")
                self.eval_loader = kwargs.get("eval_loader", None)
                self.test_loader = kwargs.get("test_loader", None)
            else:
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True
                ) if train_dataset else None
                self.eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=batch_size
                ) if eval_dataset else None
                self.test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size
                ) if test_dataset else None

        # Call Accelerator.prepare to wrap the model and optimizers for multi-GPU or multi-device support
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

        self.predictions = {}


    def evaluate(self):
        self.model.eval()
        all_truth = []
        all_preds = []

        # Use tqdm for progress bar and disable it in non-main processes
        it = tqdm(self.eval_loader, desc="Evaluating", disable=not self.accelerator.is_main_process)

        with torch.no_grad():
            for batch in it:
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                # Gather predictions and labels from all processes
                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                # Only in the main process, collect the predictions and labels
                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.cpu().numpy(force=True)
                    gathered_labels = gathered_labels.cpu().numpy(force=True)
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # self.accelerator.wait_for_everyone()

        # Only in the main process, compute metrics
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)

            valid_metrics = {}
            for metric_func in self.compute_metrics:
                valid_metrics.update(metric_func(all_truth, all_preds))

            fprint(valid_metrics)
        else:
            valid_metrics = None

        return valid_metrics

    def unwrap_model(self, model):
        """
        Unwrap the model from a distributed training wrapper (if needed).
        
        Parameters:
            model (nn.Module): The model to unwrap.
            
        Returns:
            nn.Module: The unwrapped model.
        """
        try:
            return self.accelerator.unwrap_model(model)
        except:
            try:
                return model.module
            except:
                return model

    def test(self):
        """
        Test the model on the test dataset.

        Returns:
            dict: A dictionary of test metrics.
        """
        self.model.eval()
        all_truth = []
        all_preds = []

        it = tqdm(self.test_loader, desc="Testing", disable=not self.accelerator.is_main_process)

        with torch.no_grad():
            for batch in it:
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.cpu().numpy(force=True)
                    gathered_labels = gathered_labels.cpu().numpy(force=True)
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # self.accelerator.wait_for_everyone()

        # Only in the main process, compute metrics
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)

            test_metrics = {}
            for metric_func in self.compute_metrics:
                test_metrics.update(metric_func(all_truth, all_preds))

            fprint(test_metrics)
        else:
            test_metrics = None

        return test_metrics

    def train(self, path_to_save=None, **kwargs):
        """
        Train the model, evaluate periodically, and save the best model.

        Parameters:
            path_to_save (str, optional): Path to save the model checkpoints.
            **kwargs: Additional arguments for saving the model.
        """
        seed_everything(self.seed)
        # # Initialize early stopping flag
        early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Ensure synchronization of all processes before starting
        self.accelerator.wait_for_everyone()

        # Initial validation or test
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            valid_metrics = self.evaluate()
        else:
            valid_metrics = self.test()

        # In the main process, update metrics and save the model if necessary
        if self.accelerator.is_main_process:
            if self._is_metric_better(valid_metrics, stage="valid"):
                self._save_state_dict()
                early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Synchronize early stop flags across all processes
        gathered_flags = self.accelerator.gather(early_stop_flag)
        early_stop_flag = gathered_flags if gathered_flags.ndim==0 else gathered_flags[0] 

        for epoch in range(self.epochs):
            self.model.train()

            train_it = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Loss",
                            disable=not self.accelerator.is_main_process)
            train_it = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs} Loss",
                disable=not self.accelerator.is_main_process,
            )

            # Use accelerator.accumulate to control gradient accumulation
            for step, batch in enumerate(train_it):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                    self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Synchronize all processes before evaluation
            self.accelerator.wait_for_everyone()

            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            # Update metrics and check early stopping condition in the main process
            if self.accelerator.is_main_process:
                if self._is_metric_better(valid_metrics, stage="valid"):
                    self._save_state_dict()
                    early_stop_flag = torch.tensor(0, device=self.accelerator.device)
                else:
                    early_stop_flag += 1

            # Synchronize early stop flag across all processes
            gathered_flags = self.accelerator.gather(early_stop_flag)
            early_stop_flag = gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]  # Use the main process value


            # Check if early stopping is needed
            if early_stop_flag.item() > self.patience:
                if self.accelerator.is_main_process:
                    print(f"Early stopping at epoch {epoch + 1}.")
                    fprint(f"Early stopping at epoch {epoch + 1}.")
                break

            # Save the checkpoint only in the main process
            if path_to_save and self.accelerator.is_main_process:
                _path_to_save = path_to_save + "_epoch_" + str(epoch + 1)
                if valid_metrics:
                    for key, value in valid_metrics.items():
                        _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
                self.save_model(_path_to_save, **kwargs)

            # Ensure all processes synchronize before the next epoch
            self.accelerator.wait_for_everyone()

        # Final test using the best checkpoint
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            self.accelerator.wait_for_everyone()  # Ensure loading is complete before testing
            test_metrics = self.test()
            if self.accelerator.is_main_process:
                self._is_metric_better(test_metrics, stage="test")

        # Save the final model only in the main process
        if path_to_save and self.accelerator.is_main_process:
            _path_to_save = path_to_save + "_final"
            if self.metrics.get("test"):
                for key, value in self.metrics["test"][-1].items():
                    _path_to_save += f"_seed_{self.seed}_{key}_{value:.4f}"
            self.save_model(_path_to_save, **kwargs)

        self._remove_state_dict()

        # Free accelerator memory
        self.accelerator.free_memory()
        del (
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.test_loader,
        )

        return self.metrics

    def _is_metric_better(self, metrics, stage="valid"):
        # Only compare metrics in the main process
        if not self.accelerator.is_main_process:
            return False

        assert stage in ["valid", "test"], "The metrics stage should be either 'valid' or 'test'."
        assert stage in [
            "valid",
            "test",
        ], "The metrics stage should be either 'valid' or 'test'."

        prev_metrics = self.metrics.get(stage, None)
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
        
        # Compare metrics for optimization direction
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


    def predict(self, data_loader):
        return self.accelerator.unwrap_model(self.model).predict(data_loader)

    def get_model(self, **kwargs):
        return self.model

    def compute_metrics(self):
        raise NotImplementedError(
            "The compute_metrics() function should be implemented for your model."
            " It should return a dictionary of metrics."
        )

    def save_model(self, path, overwrite=False, **kwargs):
        # Make certain only one process saves, if you're in distributed mode
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save(path, overwrite, **kwargs)

    def _load_state_dict(self):
        if hasattr(self, "_model_state_dict_path") and os.path.exists(
            self._model_state_dict_path
        ):
            weights = torch.load(self._model_state_dict_path, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(weights)

    def _save_state_dict(self):
        # Generate a unique path for saving state dict
        if not hasattr(self, "_model_state_dict_path"):
            from hashlib import sha256

            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            hash_digest = sha256(self.__repr__().encode("utf-8")).hexdigest()
            self._model_state_dict_path = f"tmp_ckpt_{time_str}_{hash_digest}.pt"

        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)

        # Use accelerator to gather model weights on one process
        if self.accelerator.is_main_process:
            torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self._model_state_dict_path)
            torch.save(
                self.accelerator.unwrap_model(self.model).state_dict(), self._model_state_dict_path
            )

    def _remove_state_dict(self):
        # Remove the model's state dict after training is complete
        if not hasattr(self, "_model_state_dict_path"):
            from hashlib import sha256

            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            hash_digest = sha256(self.__repr__().encode("utf-8")).hexdigest()
            self._model_state_dict_path = f"tmp_ckpt_{time_str}_{hash_digest}.pt"

        if (
            os.path.exists(self._model_state_dict_path)
            and self.accelerator.is_main_process
        ):
            os.remove(self._model_state_dict_path)

