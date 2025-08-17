# -*- coding: utf-8 -*-
# file: accelerate_trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Accelerate-based distributed training utilities.

This module provides HuggingFace Accelerate-based distributed training framework
for genomic models, including automatic mixed precision training, distributed
training support, early stopping, and model checkpointing.
"""

import os
import time
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Union

import torch

from .base_trainer import BaseTrainer
from ..misc.utils import env_meta_info, fprint, seed_everything


class AccelerateTrainer(BaseTrainer):
    """
    HuggingFace Accelerate-based distributed trainer for genomic models.

    This trainer provides distributed training capabilities with automatic mixed precision,
    gradient accumulation, and early stopping. It supports both single and multi-GPU
    training with seamless integration with HuggingFace Accelerate.

    Attributes:
        accelerator: HuggingFace Accelerate instance for distributed training
        early_stop_flag: Tensor for coordinating early stopping across processes

    Example:
        >>> trainer = AccelerateTrainer(
        ...     model=model,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     epochs=10,
        ...     batch_size=32,
        ...     optimizer=optimizer
        ... )
        >>> metrics = trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        **kwargs,
    ):
        """
        Initialize the accelerate trainer.

        Args:
            model (torch.nn.Module): The model to be trained
            **kwargs: Additional keyword arguments passed to BaseTrainer
        """
        super().__init__(model, **kwargs)

    def _setup_training_components(self) -> None:
        """
        Set up accelerate training-specific components.

        This method initializes the HuggingFace Accelerator with appropriate
        mixed precision settings and prepares the model, optimizer, and
        data loaders for distributed training.
        """
        # Determine mixed precision from `autocast` argument
        if self.autocast in ["float16", "fp16"]:
            mp_setting = "fp16"
        elif self.autocast in ["bfloat16", "bf16"]:
            mp_setting = "bf16"
        else:
            mp_setting = "no"

        # Initialize Accelerator
        from accelerate import Accelerator
        from accelerate import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision=mp_setting, kwargs_handlers=[ddp_kwargs]
        )

        # Prepare components with accelerator
        to_prepare = [self.model]
        if self.optimizer is not None:
            to_prepare.append(self.optimizer)
        if self.train_loader is not None:
            to_prepare.append(self.train_loader)
        if self.eval_loader is not None:
            to_prepare.append(self.eval_loader)
        if self.test_loader is not None:
            to_prepare.append(self.test_loader)

        prepared = self.accelerator.prepare(*to_prepare)
        self.model = prepared[0]

        idx = 1
        if self.optimizer is not None:
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

    def _prepare_batch(self, batch: Any) -> Any:
        """
        Prepare a batch for model input.

        For accelerate trainer, the batch is already prepared by accelerator,
        so we just return it as-is.

        Args:
            batch: Input batch

        Returns:
            The input batch (already prepared by accelerator)
        """
        return batch

    def _predict_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for a batch using the model.

        Args:
            batch: Input batch

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predictions
        """
        return self.accelerator.unwrap_model(self.model).predict(batch)

    def _train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch using HuggingFace Accelerate.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        train_loss = []

        train_it = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.epochs} Loss",
            disable=not self.accelerator.is_main_process,
        )

        # Training loop with accelerator.accumulate for gradient accumulation
        for step, batch in enumerate(train_it):
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = self._compute_loss(outputs)

                # Backward pass and optimizer step within accumulate context
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_loss.append(loss.item())
            train_it.set_description(
                f"Epoch {epoch + 1}/{self.epochs} Loss: {np.nanmean(train_loss):.4f}"
            )

        return np.nanmean(train_loss)

    def _compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss from model outputs.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs

        Returns:
            torch.Tensor: Computed loss

        Raises:
            ValueError: If no loss function is available
        """
        if "loss" in outputs:
            return outputs["loss"]

        # Try to use model's loss function
        if hasattr(self.model, "loss_function") and callable(self.model.loss_function):
            return self.model.loss_function(outputs["logits"], outputs["labels"])

        if (
            hasattr(self.model, "model")
            and hasattr(self.model.model, "loss_function")
            and callable(self.model.model.loss_function)
        ):
            return self.model.model.loss_function(outputs["logits"], outputs["labels"])

        raise ValueError(
            "The model does not have a loss function defined. "
            "Please provide a loss function or ensure the model has one."
        )

    def train(self, path_to_save: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model using distributed training.

        This method performs the complete training loop with validation,
        early stopping, and model checkpointing. It handles distributed
        training across multiple GPUs and processes.

        Args:
            path_to_save (Optional[str]): Path to save the trained model
            **kwargs: Additional keyword arguments for model saving

        Returns:
            Dict[str, Any]: Dictionary containing training metrics
        """
        seed_everything(self.seed)

        # Initialize early stopping flag for distributed coordination
        self.early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Ensure all processes sync before starting
        self.accelerator.wait_for_everyone()

        # Initial evaluation
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            initial_metrics = self.evaluate()
        else:
            initial_metrics = self.test()

        # Only main process handles metric comparison and model saving
        if self.accelerator.is_main_process:
            if self._is_metric_better(initial_metrics, stage="valid"):
                self._save_state_dict()
                self.early_stop_flag = torch.tensor(0, device=self.accelerator.device)

        # Synchronize early stopping flag across all processes
        gathered_flags = self.accelerator.gather(self.early_stop_flag)
        self.early_stop_flag = (
            gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
        )

        # Main training loop
        for epoch in range(self.epochs):
            # Train for one epoch
            avg_loss = self._train_epoch(epoch)

            # Synchronize all processes before evaluation
            self.accelerator.wait_for_everyone()

            # Evaluate after each epoch
            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            # Only main process handles metric comparison and early stopping
            if self.accelerator.is_main_process:
                if self._is_metric_better(valid_metrics, stage="valid"):
                    self._save_state_dict()
                    self.early_stop_flag = torch.tensor(
                        0, device=self.accelerator.device
                    )
                else:
                    self.early_stop_flag += 1

            # Synchronize early stopping flag across all processes
            gathered_flags = self.accelerator.gather(self.early_stop_flag)
            self.early_stop_flag = (
                gathered_flags if gathered_flags.ndim == 0 else gathered_flags[0]
            )

            # Check for early stopping
            if self.early_stop_flag.item() > self.patience:
                if self.accelerator.is_main_process:
                    fprint(f"Early stopping at epoch {epoch + 1}.")
                break

            # Save epoch checkpoint (only main process)
            if path_to_save and self.accelerator.is_main_process:
                self._save_epoch_checkpoint(
                    path_to_save, epoch, valid_metrics, **kwargs
                )

            # Ensure all processes sync before next epoch
            self.accelerator.wait_for_everyone()

        # Final testing with best model
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            self.accelerator.wait_for_everyone()
            test_metrics = self.test()
            if self.accelerator.is_main_process:
                self._is_metric_better(test_metrics, stage="test")

        # Save final model (only main process)
        if path_to_save and self.accelerator.is_main_process:
            self._save_final_model(path_to_save, **kwargs)

        # Clean up
        self._remove_state_dict()
        ## DO NOT REMOVE THE FOLLOWING BLOCK, OTHERWISE AUTOBENCH WILL NOT WORK ##
        self.accelerator.free_memory(
            self.model,
            self.optimizer,
            self.train_loader,
            self.eval_loader,
            self.test_loader,
        )
        ## Remove accelerator reference to avoid memory leaks ##
        # delattr(self, "accelerator")
        return self.metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        This method runs the model in evaluation mode and computes metrics
        on the validation dataset. It handles distributed evaluation and
        gathers results from all processes.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_truth = []
        all_preds = []

        # Disable progress bar on non-main processes
        it = tqdm(
            self.eval_loader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        )

        with torch.no_grad():
            for batch in it:
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                # Gather predictions and labels from all processes
                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                # Only main process processes gathered data
                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.float().cpu().numpy()
                    gathered_labels = gathered_labels.float().cpu().numpy()
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # Only main process computes metrics
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

            fprint(valid_metrics)
        else:
            valid_metrics = None

        self.predictions.update({"valid": {"pred": all_preds, "true": all_truth}})
        return valid_metrics

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the test dataset.

        This method runs the model in evaluation mode and computes metrics
        on the test dataset. It handles distributed testing and gathers
        results from all processes.

        Returns:
            Dict[str, Any]: Dictionary containing test metrics
        """
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
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)

                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.float().cpu().numpy()
                    gathered_labels = gathered_labels.float().cpu().numpy()
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)

        # Only main process computes metrics
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
            fprint(test_metrics)
        else:
            test_metrics = None

        self.predictions.update({"test": {"pred": all_preds, "true": all_truth}})
        return test_metrics

    def _is_metric_better(self, metrics: Dict[str, Any], stage: str = "valid") -> bool:
        """
        Check if the current metrics are better than the best metrics so far.

        This method is overridden to ensure only the main process performs
        metric comparison in distributed training.

        Args:
            metrics (Dict[str, Any]): Current metrics
            stage (str): Stage of evaluation ('valid' or 'test')

        Returns:
            bool: True if current metrics are better, False otherwise
        """
        # Only main process performs metric comparison
        if not self.accelerator.is_main_process:
            return False

        return super()._is_metric_better(metrics, stage)

    def save_model(self, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
            overwrite (bool): Whether to overwrite existing files (default: False)
            **kwargs: Additional keyword arguments for model saving
        """
        # Only main process saves the model
        if not hasattr(self, "accelerator"):
            self.model.save(path, overwrite, **kwargs)
        elif self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.model).save(path, overwrite, **kwargs)

    def _load_state_dict(self) -> None:
        """Load the best model state dictionary."""
        if hasattr(self, "_model_state_dict_path") and os.path.exists(
            self._model_state_dict_path
        ):
            weights = torch.load(self._model_state_dict_path, map_location="cpu")
            self.accelerator.unwrap_model(self.model).load_state_dict(weights)

    def _save_state_dict(self) -> None:
        """Save the current model state dictionary."""
        if not hasattr(self, "_model_state_dict_path"):
            from hashlib import sha256

            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            hash_digest = sha256(self.__repr__().encode("utf-8")).hexdigest()
            self._model_state_dict_path = f"tmp_ckpt_{time_str}_{hash_digest}.pt"

        if os.path.exists(self._model_state_dict_path):
            os.remove(self._model_state_dict_path)

        # Only main process saves the state dict
        if self.accelerator.is_main_process:
            torch.save(
                self.accelerator.unwrap_model(self.model).state_dict(),
                self._model_state_dict_path,
            )

    def _remove_state_dict(self) -> None:
        """Remove the temporary model state dictionary file."""
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
