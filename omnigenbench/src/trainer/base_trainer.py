# -*- coding: utf-8 -*-
# file: base_trainer.py
# time: 15:00 15/07/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
"""
Base trainer class.

This module provides the abstract base class for all trainers in the OmniGenome
framework. It defines the common interface and shared functionality that all
trainer implementations should provide.
"""

import os
import warnings
import tempfile
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch

from ..misc.utils import env_meta_info, fprint, seed_everything


class MetricsDict(dict):
    """
    A custom dictionary class for storing training metrics with enhanced readability.

    This class extends the built-in dict to provide a more readable string representation
    of metrics, making it easier to understand training progress and results.
    """

    def __repr__(self) -> str:
        """
        Return a formatted, human-readable string representation of the metrics.

        Returns:
            str: A formatted string showing metrics organized by stage (train/valid/test)
                 and best validation metrics.
        """
        if not self:
            return "MetricsDict(empty)"

        lines = ["=" * 80]
        lines.append("Training Metrics Summary".center(80))
        lines.append("=" * 80)

        # Display best validation metrics first if available
        if "best_valid" in self:
            lines.append("\n[Best Validation Metrics]")
            lines.append("-" * 80)
            best_metrics = self["best_valid"]
            if isinstance(best_metrics, dict):
                for key, value in best_metrics.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key:.<40} {value:.6f}")
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        if len(value) > 0 and isinstance(value[0], (int, float)):
                            mean_val = np.mean(value)
                            std_val = np.std(value)
                            lines.append(f"  {key:.<40} {mean_val:.6f} ± {std_val:.6f}")
                        else:
                            lines.append(f"  {key:.<40} {value}")
                    else:
                        lines.append(f"  {key:.<40} {value}")

        # Display metrics for each stage (train, valid, test)
        for stage in ["train", "valid", "test"]:
            if stage in self and self[stage]:
                lines.append(f"\n[{stage.capitalize()} Metrics History]")
                lines.append("-" * 80)

                metrics_list = self[stage]
                if isinstance(metrics_list, list) and len(metrics_list) > 0:
                    # Show the number of epochs/evaluations
                    lines.append(f"  Total evaluations: {len(metrics_list)}")

                    # Show the latest metrics
                    latest_metrics = metrics_list[-1]
                    if isinstance(latest_metrics, dict):
                        lines.append(f"  Latest (Epoch {len(metrics_list)}):")
                        for key, value in latest_metrics.items():
                            if isinstance(value, (int, float)):
                                lines.append(f"    {key:.<38} {value:.6f}")
                            elif isinstance(value, (list, tuple, np.ndarray)):
                                if len(value) > 0 and isinstance(
                                    value[0], (int, float)
                                ):
                                    mean_val = np.mean(value)
                                    std_val = np.std(value)
                                    lines.append(
                                        f"    {key:.<38} {mean_val:.6f} ± {std_val:.6f}"
                                    )
                                else:
                                    lines.append(f"    {key:.<38} {value}")
                            else:
                                lines.append(f"    {key:.<38} {value}")

                    # Show trend if we have multiple epochs
                    if len(metrics_list) > 1:
                        lines.append(f"  First (Epoch 1):")
                        first_metrics = metrics_list[0]
                        if isinstance(first_metrics, dict):
                            for key, value in first_metrics.items():
                                if isinstance(value, (int, float)):
                                    lines.append(f"    {key:.<38} {value:.6f}")
                                elif isinstance(value, (list, tuple, np.ndarray)):
                                    if len(value) > 0 and isinstance(
                                        value[0], (int, float)
                                    ):
                                        mean_val = np.mean(value)
                                        lines.append(f"    {key:.<38} {mean_val:.6f}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the same formatted representation as __repr__."""
        return self.__repr__()


def _infer_optimization_direction(
    metrics: Dict[str, Any], prev_metrics: List[Dict[str, Any]]
) -> str:
    """
    Infer the optimization direction based on metric values.

    This function analyzes the trend of metric values to determine whether
    larger values are better (e.g., accuracy) or smaller values are better
    (e.g., loss).

    Args:
        metrics (Dict[str, Any]): Current metric values
        prev_metrics (List[Dict[str, Any]]): Previous metric values from multiple epochs

    Returns:
        str: Either 'larger_is_better' or 'smaller_is_better'
    """
    # Check if metrics is empty
    if not metrics or len(metrics) == 0:
        fprint(
            "Warning: Cannot infer optimization direction from empty metrics. Defaulting to 'smaller_is_better'."
        )
        return "smaller_is_better"

    larger_is_better_metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
        "roc_auc",
        "pr_auc",
        "score",
        "auc",
        "balanced_accuracy",
        "matthews_corrcoef",
        "jaccard",
        "dice",
    ]
    smaller_is_better_metrics = [
        "loss",
        "error",
        "mse",
        "mae",
        "rmse",
        "r2",
        "distance",
        "perplexity",
        "cross_entropy",
        "binary_cross_entropy",
        "focal_loss",
        "huber_loss",
    ]

    # Check if any metric name matches known patterns
    for metric_name in metrics.keys():
        metric_name_lower = metric_name.lower()
        for pattern in larger_is_better_metrics:
            if pattern in metric_name_lower:
                return "larger_is_better"
        for pattern in smaller_is_better_metrics:
            if pattern in metric_name_lower:
                return "smaller_is_better"

    # If no pattern matches, try to infer from metric trends
    if prev_metrics and len(prev_metrics) >= 2:
        fprint(
            "Cannot determine optimization direction from metric names. Attempting inference from trends."
        )

        try:
            # Check if prev_metrics have non-empty values
            if (
                not list(metrics.values())
                or not list(prev_metrics[-1].values())
                or not list(prev_metrics[0].values())
            ):
                fprint("Warning: Empty metric values found. Cannot infer from trends.")
                return "smaller_is_better"

            # Get the first metric value for trend analysis
            first_metric_key = list(metrics.keys())[0]
            current_value = np.mean(
                list(metrics.values())[0]
                if isinstance(list(metrics.values())[0], (list, tuple, np.ndarray))
                else [list(metrics.values())[0]]
            )
            prev_value = np.mean(
                list(prev_metrics[-1].values())[0]
                if isinstance(
                    list(prev_metrics[-1].values())[0], (list, tuple, np.ndarray)
                )
                else [list(prev_metrics[-1].values())[0]]
            )
            earlier_value = np.mean(
                list(prev_metrics[0].values())[0]
                if isinstance(
                    list(prev_metrics[0].values())[0], (list, tuple, np.ndarray)
                )
                else [list(prev_metrics[0].values())[0]]
            )

            # Check if metrics are consistently increasing or decreasing
            is_increasing = earlier_value < prev_value < current_value
            is_decreasing = earlier_value > prev_value > current_value

            if is_increasing:
                return "larger_is_better"
            elif is_decreasing:
                return "smaller_is_better"
        except (IndexError, KeyError, TypeError) as e:
            fprint(f"Error inferring optimization direction: {e}")

    # Default to smaller_is_better (common for loss-based metrics)
    fprint(
        "Cannot determine optimization direction. Defaulting to 'smaller_is_better'."
    )
    return "smaller_is_better"


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers in the OmniGenome framework.

    This class defines the common interface and shared functionality that all
    trainer implementations should provide. It includes methods for training,
    evaluation, testing, and model management.

    Attributes:
        model: The model to be trained
        train_loader: DataLoader for training data
        eval_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        gradient_accumulation_steps: Number of steps for gradient accumulation
        optimizer: Optimizer for training
        loss_fn: Loss function
        compute_metrics: List of metric computation functions
        seed: Random seed for reproducibility
        metrics: Dictionary to store training metrics
        predictions: Dictionary to store model predictions
        metadata: Dictionary containing environment and training metadata
        trial_name: Name of the current training trial
        _optimization_direction: Optimization direction ('larger_is_better' or 'smaller_is_better')

    Example:
        >>> class MyTrainer(BaseTrainer):
        ...     def _setup_training_components(self):
        ...         # Implementation specific setup
        ...         pass
        ...     def _train_epoch(self, epoch):
        ...         # Implementation specific training loop
        ...         pass
        >>> trainer = MyTrainer(model=model, train_dataset=train_dataset)
        >>> metrics = trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        patience: int = -1,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[torch.nn.Module] = None,
        compute_metrics: Optional[Union[List, str]] = None,
        seed: int = 42,
        autocast: str = "float16",
        **kwargs,
    ):
        """
        Initialize the base trainer.

        Args:
            model (torch.nn.Module): The model to be trained
            train_dataset (Optional[Dataset]): Training dataset
            eval_dataset (Optional[Dataset]): Validation dataset
            test_dataset (Optional[Dataset]): Test dataset
            epochs (int): Number of training epochs (default: 3)
            batch_size (int): Batch size for training (default: 8)
            patience (int): Early stopping patience (default: -1, no early stopping)
            gradient_accumulation_steps (int): Gradient accumulation steps (default: 1)
            optimizer (Optional[torch.optim.Optimizer]): Optimizer for training
            loss_fn (Optional[torch.nn.Module]): Loss function
            compute_metrics (Optional[Union[List, str]]): Metric computation functions
            seed (int): Random seed for reproducibility (default: 42)
            autocast (str): Mixed precision type (default: "float16")
            **kwargs: Additional keyword arguments
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience if patience > 0 else epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optimizer = optimizer
        if not optimizer:
            warnings.warn(
                f"No optimizer provided. Defaulting to Adam optimizer with {learning_rate}."
            )
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # Set loss function if provided
        if loss_fn is not None and hasattr(self.model, "set_loss_fn"):
            self.model.set_loss_fn(loss_fn)
        self.compute_metrics = (
            (
                compute_metrics
                if isinstance(compute_metrics, list)
                else [compute_metrics]
            )
            if compute_metrics
            else []
        )
        if not self.compute_metrics:
            warnings.warn(
                "No compute metrics provided. Metrics will not be calculated during training."
            )
        self.seed = seed
        seed_everything(seed)
        self.autocast = autocast

        # Initialize data loaders
        self._setup_data_loaders(
            train_dataset, eval_dataset, test_dataset, batch_size, **kwargs
        )

        # Initialize training components
        self._setup_training_components()

        # Initialize metadata and tracking
        self.metadata = env_meta_info()
        self.metrics = MetricsDict()
        self.predictions = {}
        self._optimization_direction = None
        self.trial_name = kwargs.get("trial_name", self.model.__class__.__name__)

        if not hasattr(self, "_model_state_dict_path"):
            # Create temporary directory if it doesn't exist
            temp_dir = tempfile.gettempdir()
            os.makedirs(temp_dir, exist_ok=True)

            # Use mkstemp for better cross-platform compatibility
            fd, temp_path = tempfile.mkstemp(suffix=".pt", dir=temp_dir)
            os.close(fd)  # Close file descriptor immediately
            self._model_state_dict_path = temp_path

    def _setup_data_loaders(
        self,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Dataset],
        test_dataset: Optional[Dataset],
        batch_size: int,
        **kwargs,
    ) -> None:
        """
        Set up data loaders for training, evaluation, and testing.

        Args:
            train_dataset (Optional[Dataset]): Training dataset
            eval_dataset (Optional[Dataset]): Validation dataset
            test_dataset (Optional[Dataset]): Test dataset
            batch_size (int): Batch size for data loaders
            **kwargs: Additional keyword arguments
        """
        # Check if pre-built loaders are provided
        if kwargs.get("train_loader"):
            self.train_loader = kwargs.get("train_loader")
        else:
            if train_dataset is None:
                raise ValueError(
                    "train_dataset must be provided if train_loader is not."
                )
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
        if kwargs.get("eval_loader") or kwargs.get("valid_loader"):
            self.eval_loader = kwargs.get("eval_loader", None) or kwargs.get(
                "valid_loader", None
            )
        else:
            # Create data loaders from datasets
            self.eval_loader = (
                DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
                if eval_dataset is not None
                else None
            )
        if kwargs.get("test_loader"):
            self.test_loader = kwargs.get("test_loader", None)
        else:
            self.test_loader = (
                DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                if test_dataset is not None
                else None
            )

    @abstractmethod
    def _setup_training_components(self) -> None:
        """
        Set up training-specific components (device, scaler, etc.).

        This method should be implemented by subclasses to initialize
        trainer-specific components like device selection, gradient scalers,
        distributed training setup, etc.
        """
        pass

    @abstractmethod
    def _train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number

        Returns:
            float: Average training loss for the epoch
        """
        pass

    def _is_metric_better(self, metrics: Dict[str, Any], stage: str = "valid") -> bool:
        """
        Check if the current metrics are better than the best metrics so far.

        Args:
            metrics (Dict[str, Any]): Current metric values
            stage (str): Stage name ("valid" or "test")

        Returns:
            bool: True if current metrics are better than best metrics

        Raises:
            AssertionError: If stage is not "valid" or "test"
        """
        assert stage in [
            "valid",
            "test",
        ], "The metrics stage should be either 'valid' or 'test'."

        # Check if metrics is empty
        if not metrics or len(metrics) == 0:
            fprint("Warning: Empty metrics dictionary received. Skipping comparison.")
            # Still store the empty metrics for tracking
            if stage not in self.metrics:
                self.metrics[stage] = [metrics]
            else:
                self.metrics[stage].append(metrics)
            return False

        # Store current metrics
        prev_metrics = self.metrics.get(stage, None)
        if stage not in self.metrics:
            self.metrics[stage] = [metrics]
        else:
            self.metrics[stage].append(metrics)

        # Initialize best metrics if not present (only with non-empty metrics)
        if "best_valid" not in self.metrics:
            self.metrics["best_valid"] = metrics
            return True

        # If we have best_valid but prev_metrics is None, this is the second call
        # We should still compare against best_valid
        if prev_metrics is None or len(prev_metrics) == 0:
            prev_metrics = [self.metrics.get("best_valid", {})]

        # Determine optimization direction
        if self._optimization_direction is None:
            self._optimization_direction = _infer_optimization_direction(
                metrics, prev_metrics
            )

        # Compare metrics based on optimization direction
        try:
            # Check if metrics has values before accessing
            if not list(metrics.values()):
                fprint(
                    "Warning: Metrics dictionary has no values. Skipping comparison."
                )
                return False

            current_value = np.mean(
                list(metrics.values())[0]
                if isinstance(list(metrics.values())[0], (list, tuple, np.ndarray))
                else [list(metrics.values())[0]]
            )

            # Check if best_valid has values
            if not list(self.metrics["best_valid"].values()):
                fprint(
                    "Warning: Best metrics dictionary has no values. Skipping comparison."
                )
                return False

            best_value = np.mean(
                list(self.metrics["best_valid"].values())[0]
                if isinstance(
                    list(self.metrics["best_valid"].values())[0],
                    (list, tuple, np.ndarray),
                )
                else [list(self.metrics["best_valid"].values())[0]]
            )

            if self._optimization_direction == "larger_is_better":
                is_better = current_value > best_value
            else:  # smaller_is_better
                is_better = current_value < best_value

            if is_better:
                self.metrics["best_valid"] = metrics
                return True
        except (IndexError, KeyError, TypeError) as e:
            fprint(f"Error comparing metrics: {e}")
            return False

        return False

    def train(self, path_to_save: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model.

        This method implements the main training loop including early stopping,
        model checkpointing, and metric tracking.

        Args:
            path_to_save (Optional[str]): Path to save the trained model
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Any]: Training metrics and results
        """
        seed_everything(self.seed)
        patience_counter = 0

        # Initial evaluation
        if self.eval_loader is not None and len(self.eval_loader) > 0:
            initial_metrics = self.evaluate()
        else:
            initial_metrics = self.test()

        if self._is_metric_better(initial_metrics, stage="valid"):
            self._save_state_dict()
            patience_counter = 0

        # Main training loop
        for epoch in range(self.epochs):
            # Train for one epoch
            avg_loss = self._train_epoch(epoch)

            # Evaluate after each epoch
            if self.eval_loader is not None and len(self.eval_loader) > 0:
                valid_metrics = self.evaluate()
            else:
                valid_metrics = self.test()

            # Check for improvement and early stopping
            if self._is_metric_better(valid_metrics, stage="valid"):
                self._save_state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    fprint(f"Early stopping at epoch {epoch + 1}.")
                    break

            # Save epoch checkpoint if requested
            if path_to_save:
                self._save_epoch_checkpoint(
                    path_to_save, epoch, valid_metrics, **kwargs
                )

        # Final testing with best model
        if self.test_loader is not None and len(self.test_loader) > 0:
            self._load_state_dict()
            test_metrics = self.test()
            self._is_metric_better(test_metrics, stage="test")

        # Save final model if requested
        if path_to_save:
            self._save_final_model(path_to_save, **kwargs)

        # Clean up temporary files
        self._remove_state_dict()

        return self.metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_truth = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = self._prepare_batch(batch)
                output = self._predict_batch(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                all_truth.append(self._process_labels(labels))
                all_preds.append(self._process_predictions(predictions))

        # Concatenate all predictions and labels
        all_truth = self._concatenate_outputs(all_truth)
        all_preds = self._concatenate_outputs(all_preds)

        # Compute metrics
        if not np.all(all_truth == -100):
            valid_metrics = {}
            for metric_func in self.compute_metrics:
                valid_metrics.update(metric_func(all_truth, all_preds))
            fprint(valid_metrics)
        else:
            valid_metrics = {"Validation labels may be NaN. No metrics calculated.": 0}

        # Store predictions
        self.predictions.update({"valid": {"pred": all_preds, "true": all_truth}})

        return valid_metrics

    def test(self) -> Dict[str, Any]:
        """
        Test the model on the test dataset.

        Returns:
            Dict[str, Any]: Dictionary containing test metrics
        """
        self.model.eval()
        all_truth = []
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = self._prepare_batch(batch)
                output = self._predict_batch(batch)
                predictions = output["predictions"]
                labels = batch["labels"]

                all_truth.append(self._process_labels(labels))
                all_preds.append(self._process_predictions(predictions))

        # Concatenate all predictions and labels
        all_truth = self._concatenate_outputs(all_truth)
        all_preds = self._concatenate_outputs(all_preds)

        # Compute metrics
        if not np.all(all_truth == -100):
            test_metrics = {}
            for metric_func in self.compute_metrics:
                test_metrics.update(metric_func(all_truth, all_preds))
            fprint(test_metrics)
        else:
            test_metrics = {"Test labels may be NaN. No metrics calculated.": 0}

        # Store predictions
        self.predictions.update({"test": {"pred": all_preds, "true": all_truth}})

        return test_metrics

    def predict(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Generate predictions using the trained model.

        Args:
            data_loader (DataLoader): DataLoader for prediction data

        Returns:
            Dict[str, Any]: Dictionary containing predictions
        """
        return self._predict_batch(data_loader)

    def get_model(self, **kwargs) -> torch.nn.Module:
        """
        Get the trained model.

        Args:
            **kwargs: Additional keyword arguments

        Returns:
            torch.nn.Module: The trained model
        """
        return self.model

    def save_model(self, path: str, overwrite: bool = False, **kwargs) -> None:
        """
        Save the trained model.

        Args:
            path (str): Path to save the model
            overwrite (bool): Whether to overwrite existing files (default: False)
            **kwargs: Additional keyword arguments
        """
        if hasattr(self.model, "save"):
            self.model.save(path, overwrite, **kwargs)
        else:
            torch.save(self.model.state_dict(), f"{path}.pt")

    # Abstract methods that subclasses may need to implement
    @abstractmethod
    def _prepare_batch(self, batch: Any) -> Any:
        """
        Prepare a batch for model input.

        Args:
            batch: Input batch

        Returns:
            Prepared batch
        """
        pass

    @abstractmethod
    def _predict_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for a batch.

        Args:
            batch: Input batch

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing predictions
        """
        pass

    def _process_labels(self, labels: torch.Tensor) -> np.ndarray:
        """
        Process labels for metric computation.

        Args:
            labels (torch.Tensor): Raw labels

        Returns:
            np.ndarray: Processed labels
        """
        return labels.float().cpu().numpy()

    def _process_predictions(self, predictions: torch.Tensor) -> np.ndarray:
        """
        Process predictions for metric computation.

        Args:
            predictions (torch.Tensor): Raw predictions

        Returns:
            np.ndarray: Processed predictions
        """
        return predictions.float().cpu().numpy()

    def _concatenate_outputs(self, outputs: List[np.ndarray]) -> np.ndarray:
        """
        Concatenate list of outputs.

        Args:
            outputs (List[np.ndarray]): List of output arrays

        Returns:
            np.ndarray: Concatenated outputs
        """
        if not outputs:
            return np.array([])

        sample_output = outputs[0]
        if sample_output.ndim > 1:
            return np.vstack(outputs)
        else:
            return np.hstack(outputs)

    def _save_epoch_checkpoint(
        self, path_to_save: str, epoch: int, metrics: Dict[str, Any], **kwargs
    ) -> None:
        """
        Save model checkpoint after each epoch.

        Args:
            path_to_save (str): Base path for saving
            epoch (int): Current epoch number
            metrics (Dict[str, Any]): Current metrics
            **kwargs: Additional keyword arguments
        """
        checkpoint_path = f"{path_to_save}_epoch_{epoch + 1}"

        if metrics:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    checkpoint_path += f"_seed_{self.seed}_{key}_{value:.4f}"

        self.save_model(checkpoint_path, **kwargs)

    def _save_final_model(self, path_to_save: str, **kwargs) -> None:
        """
        Save the final trained model.

        Args:
            path_to_save (str): Base path for saving
            **kwargs: Additional keyword arguments
        """
        final_path = f"{path_to_save}_final"

        if self.metrics.get("test") and len(self.metrics["test"]) > 0:
            for key, value in self.metrics["test"][-1].items():
                if isinstance(value, (int, float)):
                    final_path += f"_seed_{self.seed}_{key}_{value:.4f}"

        self.save_model(final_path, **kwargs)

    def _save_state_dict(self) -> None:
        """
        Save model state dictionary to temporary file.
        """
        try:
            if os.path.exists(self._model_state_dict_path):
                os.remove(self._model_state_dict_path)
        except Exception as e:
            fprint(f"Failed to remove temporary checkpoint file: {e}")

        torch.save(self.model.state_dict(), self._model_state_dict_path)

    def _load_state_dict(self) -> None:
        """
        Load model state dictionary from temporary file.
        """
        if hasattr(self, "_model_state_dict_path") and os.path.exists(
            self._model_state_dict_path
        ):
            self.model.load_state_dict(
                torch.load(self._model_state_dict_path, map_location="cpu")
            )

    def _remove_state_dict(self) -> None:
        """
        Remove temporary state dictionary file.
        """
        if hasattr(self, "_model_state_dict_path"):
            try:
                if os.path.exists(self._model_state_dict_path):
                    os.remove(self._model_state_dict_path)
            except Exception as e:
                fprint(f"Failed to remove temporary checkpoint file: {e}")
