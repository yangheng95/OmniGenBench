# -*- coding: utf-8 -*-
# file: hf_trainer.py
# time: 14:40 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
HuggingFace trainer integration for genomic models.

This module provides HuggingFace trainer wrappers for genomic models,
enabling seamless integration with the HuggingFace training ecosystem
while maintaining OmniGenome-specific functionality.
"""

from typing import Dict, Any, Optional, Union
import torch
from transformers import Trainer
from transformers import TrainingArguments

from .base_trainer import BaseTrainer
from ... import __name__ as omnigenbench_name
from ... import __version__ as omnigenbench_version


class HFTrainer(BaseTrainer):
    """
    HuggingFace trainer wrapper for genomic models.

    This class extends the OmniGenome BaseTrainer to integrate with HuggingFace
    Trainer while maintaining OmniGenome-specific metadata and functionality.
    It provides seamless integration with the HuggingFace training ecosystem.

    Attributes:
        hf_trainer: The underlying HuggingFace Trainer instance
        training_args: HuggingFace TrainingArguments instance
        metadata: Dictionary containing OmniGenome library information

    Example:
        >>> from transformers import TrainingArguments
        >>> training_args = TrainingArguments(
        ...     output_dir="./output",
        ...     num_train_epochs=3,
        ...     per_device_train_batch_size=16,
        ... )
        >>> trainer = HFTrainer(
        ...     model=model,
        ...     train_dataset=train_dataset,
        ...     eval_dataset=eval_dataset,
        ...     training_args=training_args
        ... )
        >>> metrics = trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        training_args: Optional[TrainingArguments] = None,
        **kwargs,
    ):
        """
        Initialize the HuggingFace trainer wrapper.

        Args:
            model (torch.nn.Module): The model to be trained
            training_args (Optional[TrainingArguments]): HuggingFace training arguments
            **kwargs: Additional keyword arguments passed to BaseTrainer
        """
        # Extract training arguments or create default ones
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./output",
                num_train_epochs=kwargs.get("epochs", 3),
                per_device_train_batch_size=kwargs.get("batch_size", 8),
                per_device_eval_batch_size=kwargs.get("batch_size", 8),
                logging_dir="./logs",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )

        self.training_args = training_args

        # Initialize base trainer
        super().__init__(model, **kwargs)

        # Store metadata
        self.metadata = {
            "library_name": omnigenbench_name,
            "omnigenbench_version": omnigenbench_version,
        }

    def _setup_training_components(self) -> None:
        """
        Set up HuggingFace training-specific components.

        This method initializes the HuggingFace Trainer with the model,
        datasets, and training arguments.
        """
        # Initialize HuggingFace Trainer
        self.hf_trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_loader.dataset if self.train_loader else None,
            eval_dataset=self.eval_loader.dataset if self.eval_loader else None,
            compute_metrics=self._compute_hf_metrics if self.compute_metrics else None,
        )

    def _compute_hf_metrics(self, eval_pred):
        """
        Compute metrics for HuggingFace trainer.

        This method adapts OmniGenome metrics to work with HuggingFace trainer.

        Args:
            eval_pred: Evaluation predictions from HuggingFace trainer

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        predictions, labels = eval_pred

        # Convert to numpy arrays
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()
        if hasattr(labels, "numpy"):
            labels = labels.numpy()

        # Compute metrics using OmniGenome metric functions
        metrics = {}
        for metric_func in self.compute_metrics:
            metrics.update(metric_func(labels, predictions))

        return metrics

    def _prepare_batch(self, batch: Any) -> Any:
        """
        Prepare a batch for model input.

        For HuggingFace trainer, batch preparation is handled internally.

        Args:
            batch: Input batch

        Returns:
            The input batch
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
        return self.model.predict(batch)

    def _train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        This method is not used in HuggingFace trainer as training is handled
        by the HuggingFace Trainer.train() method.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        # This method is not used in HuggingFace trainer
        return 0.0

    def train(self, path_to_save: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model using HuggingFace Trainer.

        Args:
            path_to_save (Optional[str]): Path to save the trained model
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: Training metrics and results
        """
        # Set output directory if path_to_save is provided
        if path_to_save:
            self.training_args.output_dir = path_to_save

        # Train using HuggingFace Trainer
        train_result = self.hf_trainer.train()

        # Extract metrics
        train_metrics = train_result.metrics

        # Evaluate if eval dataset is available
        if self.eval_loader:
            eval_metrics = self.hf_trainer.evaluate()
        else:
            eval_metrics = {}

        # Test if test dataset is available
        if self.test_loader:
            test_metrics = self.test()
        else:
            test_metrics = {}

        # Combine all metrics
        all_metrics = {
            "train": train_metrics,
            "valid": eval_metrics,
            "test": test_metrics,
        }

        # Save model if requested
        if path_to_save:
            self.save_model(path_to_save, **kwargs)

        return all_metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        if self.eval_loader:
            return self.hf_trainer.evaluate()
        else:
            return {}

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the test dataset.

        Returns:
            Dict[str, Any]: Dictionary containing test metrics
        """
        if self.test_loader:
            # Use HuggingFace trainer's predict method
            predictions = self.hf_trainer.predict(self.test_loader.dataset)

            # Extract predictions and labels
            pred_labels = predictions.predictions
            true_labels = predictions.label_ids

            # Compute metrics
            if not torch.all(torch.tensor(true_labels) == -100):
                test_metrics = {}
                for metric_func in self.compute_metrics:
                    test_metrics.update(metric_func(true_labels, pred_labels))
            else:
                test_metrics = {"Test labels may be NaN. No metrics calculated.": 0}

            return test_metrics
        else:
            return {}

    def save_model(self, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
            overwrite (bool): Whether to overwrite existing files (default: False)
            **kwargs: Additional keyword arguments
        """
        # Save using HuggingFace trainer
        self.hf_trainer.save_model(path)

        # Also save using OmniGenome model's save method if available
        if hasattr(self.model, "save"):
            self.model.save(path, overwrite, **kwargs)

    def get_model(self, **kwargs) -> torch.nn.Module:
        """
        Get the trained model.

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            torch.nn.Module: The trained model
        """
        return self.hf_trainer.model


class HFTrainingArguments(TrainingArguments):
    """
    HuggingFace training arguments wrapper for genomic models.

    This class extends the HuggingFace TrainingArguments to include
    OmniGenome-specific metadata while maintaining full compatibility
    with the HuggingFace training ecosystem.

    Attributes:
        metadata: Dictionary containing OmniGenome library information

    Example:
        >>> training_args = HFTrainingArguments(
        ...     output_dir="./output",
        ...     num_train_epochs=3,
        ...     per_device_train_batch_size=16,
        ... )
        >>> trainer = HFTrainer(model=model, training_args=training_args)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the HuggingFace training arguments wrapper.

        Args:
            *args: Positional arguments passed to the parent TrainingArguments
            **kwargs: Keyword arguments passed to the parent TrainingArguments
        """
        super(HFTrainingArguments, self).__init__(*args, **kwargs)
        self.metadata = {
            "library_name": omnigenbench_name,
            "omnigenbench_version": omnigenbench_version,
        }
