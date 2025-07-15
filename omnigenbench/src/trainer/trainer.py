# -*- coding: utf-8 -*-
# file: trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
Native training utilities.

This module provides a native PyTorch training framework for genomic models,
including automatic mixed precision training, early stopping, metric tracking,
and model checkpointing.
"""
import os
import tempfile
import autocuda
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Union

import torch
from torch.cuda.amp import GradScaler

from .base_trainer import BaseTrainer
from ..misc.utils import env_meta_info, fprint, seed_everything


class Trainer(BaseTrainer):
    """
    Native PyTorch trainer for genomic models.

    This trainer provides a complete training framework with automatic mixed precision,
    early stopping, metric tracking, and model checkpointing using native PyTorch
    without distributed training dependencies.

    Attributes:
        device: Device to run training on (CPU or GPU)
        fast_dtype: Data type for mixed precision training
        scaler: Gradient scaler for mixed precision training

    Example:
        >>> trainer = Trainer(
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
        device: Optional[Union[torch.device, str]] = None,
        **kwargs,
    ):
        """
        Initialize the native trainer.

        Args:
            model (torch.nn.Module): The model to be trained
            device (Optional[Union[torch.device, str]]): Device to run training on
            **kwargs: Additional keyword arguments passed to BaseTrainer
        """
        # Set device before calling parent constructor
        self.device = device if device else autocuda.auto_cuda()
        self.device = (
            torch.device(self.device) if isinstance(self.device, str) else self.device
        )

        super().__init__(model, **kwargs)

    def _setup_training_components(self) -> None:
        """
        Set up native training-specific components.

        This method initializes the device, mixed precision settings,
        and gradient scaler for native PyTorch training.
        """
        # Set up mixed precision data type
        self.fast_dtype = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }.get(self.autocast, torch.float16)

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler()

        # Move model to device
        self.model.to(self.device)

    def _prepare_batch(self, batch: Any) -> Any:
        """
        Prepare a batch for model input by moving to device.

        Args:
            batch: Input batch

        Returns:
            Batch moved to the appropriate device
        """
        return batch.to(self.device)

    def _predict_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for a batch using the model.

        Args:
            batch: Input batch

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predictions
        """
        if self.fast_dtype and self.fast_dtype != torch.float32:
            with torch.autocast(device_type=self.device.type, dtype=self.fast_dtype):
                return self.model.predict(batch)
        else:
            return self.model.predict(batch)

    def _train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch using native PyTorch.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        train_loss = []

        train_it = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} Loss")

        for step, batch in enumerate(train_it):
            batch = self._prepare_batch(batch)

            # Zero gradients at the beginning of accumulation
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if self.fast_dtype and self.fast_dtype != torch.float32:
                with torch.autocast(
                    device_type=self.device.type, dtype=self.fast_dtype
                ):
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

            # Compute loss
            loss = self._compute_loss(outputs)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass with optional mixed precision
            if self.fast_dtype and self.fast_dtype != torch.float32:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step after accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(
                self.train_loader
            ):
                if self.fast_dtype and self.fast_dtype != torch.float32:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

            # Track loss (unscaled for display)
            train_loss.append(loss.item() * self.gradient_accumulation_steps)
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

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        with torch.no_grad():
            self.model.eval()
            val_truth = []
            val_preds = []

            it = tqdm(self.eval_loader, desc="Evaluating")
            for batch in it:
                batch = self._prepare_batch(batch)
                labels = batch["labels"]
                batch.pop("labels")

                predictions = self._predict_batch(batch)["predictions"]

                val_truth.append(labels.float().cpu().numpy())
                val_preds.append(predictions.float().cpu().numpy())

            val_truth = self._concatenate_outputs(val_truth)
            val_preds = self._concatenate_outputs(val_preds)

            if not np.all(val_truth == -100):
                valid_metrics = {}
                for metric_func in self.compute_metrics:
                    valid_metrics.update(metric_func(val_truth, val_preds))
                fprint(valid_metrics)
            else:
                valid_metrics = {
                    "Validation set labels may be NaN. No metrics calculated.": 0
                }

        self.predictions.update({"valid": {"pred": val_preds, "true": val_truth}})
        return valid_metrics

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the test dataset.

        Returns:
            Dict[str, Any]: Dictionary containing test metrics
        """
        with torch.no_grad():
            self.model.eval()
            preds = []
            truth = []

            it = tqdm(self.test_loader, desc="Testing")
            for batch in it:
                batch = self._prepare_batch(batch)
                labels = batch["labels"]
                batch.pop("labels")

                predictions = self._predict_batch(batch)["predictions"]

                truth.append(labels.float().cpu().numpy())
                preds.append(predictions.float().cpu().numpy())

            truth = self._concatenate_outputs(truth)
            preds = self._concatenate_outputs(preds)

            if not np.all(truth == -100):
                test_metrics = {}
                for metric_func in self.compute_metrics:
                    test_metrics.update(metric_func(truth, preds))
                fprint(test_metrics)
            else:
                test_metrics = {"Test set labels may be NaN. No metrics calculated.": 0}

        self.predictions.update({"test": {"pred": preds, "true": truth}})
        return test_metrics

    def unwrap_model(self, model: Optional[torch.nn.Module] = None) -> torch.nn.Module:
        """
        Unwrap the model from any distributed training wrappers.

        Args:
            model (Optional[torch.nn.Module]): Model to unwrap (default: None, uses self.model)

        Returns:
            torch.nn.Module: The unwrapped model
        """
        if model is None:
            model = self.model

        # For native trainer, no unwrapping needed typically
        try:
            return model.module  # In case of DataParallel
        except AttributeError:
            return model

    def _load_state_dict(self) -> None:
        """
        Load model state dictionary from temporary file.
        """
        if hasattr(self, "_model_state_dict_path") and os.path.exists(
            self._model_state_dict_path
        ):
            self.unwrap_model().load_state_dict(
                torch.load(self._model_state_dict_path, map_location="cpu")
            )
            self.unwrap_model().to(self.device)

    def _save_state_dict(self) -> None:
        """
        Save model state dictionary to temporary file.
        """
        if not hasattr(self, "_model_state_dict_path"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            self._model_state_dict_path = tmp_file.name
            tmp_file.close()

        try:
            if os.path.exists(self._model_state_dict_path):
                os.remove(self._model_state_dict_path)
        except Exception as e:
            fprint(
                f"Failed to remove the temporary checkpoint file {self._model_state_dict_path}: {e}"
            )

        torch.save(self.unwrap_model().state_dict(), self._model_state_dict_path)

    def save_model(self, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
            overwrite (bool): Whether to overwrite existing files (default: False)
            **kwargs: Additional keyword arguments
        """
        self.unwrap_model().save(path, overwrite, **kwargs)
