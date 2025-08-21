# -*- coding: utf-8 -*-
# file: regression_metric.py
# time: 12:57 09/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.


import types
import warnings

import numpy as np
import sklearn.metrics as metrics

from ..abc.abstract_metric import OmniMetric


def mcrmse(y_true, y_pred):
    """
    Compute Mean Column Root Mean Square Error (MCRMSE). 
    MCRMSE is a multi-target regression metric that computes the RMSE for each target
    column and then takes the mean across all targets.

    Args:
        y_true (np.ndarray): Ground truth values with shape (n_samples, n_targets)
        y_pred (np.ndarray): Predicted values with shape (n_samples, n_targets)

    Returns:
        float: Mean Column Root Mean Square Error

    Raises:
        ValueError: If y_true and y_pred have different shapes

    Example:
        >>> y_true = np.array([[1, 2], [3, 4], [5, 6]])
        >>> y_pred = np.array([[1.1, 2.1], [2.9, 4.1], [5.2, 5.8]])
        >>> mcrmse(y_true, y_pred)
        0.1833...
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = y_true != -100
    filtered_y_pred = y_pred[mask]
    filtered_y_true = y_true[mask]
    rmse_per_target = np.sqrt(np.mean((filtered_y_true - filtered_y_pred) ** 2, axis=0))
    mcrmse_value = np.mean(rmse_per_target)
    return mcrmse_value


setattr(metrics, "mcrmse", mcrmse)


class RegressionMetric(OmniMetric):
    """
    This class provides access to regression-specific metrics from scikit-learn
    and handles different input formats including HuggingFace trainer outputs.
    It dynamically wraps scikit-learn metrics and provides a unified interface
    for computing various regression evaluation metrics.

    Attributes:
        metric_func: Custom metric function if provided
        ignore_y: Value to ignore in predictions and true values
        kwargs: Additional keyword arguments for metric computation
        metrics: Dictionary of available metrics including custom ones

    Example:
        >>> from omnigenbench import RegressionMetric
        >>> metric = RegressionMetric(ignore_y=-100)
        >>> y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> y_pred = [1.1, 1.9, 3.1, 3.9, 5.2]
        >>> result = metric.mean_squared_error(y_true, y_pred)
        >>> print(result)
        {'mean_squared_error': 0.012}
    """

    def __init__(self, metric_func=None, ignore_y=-100, *args, **kwargs):
        """
        Initialize the RegressionMetric class.

        Args:
            metric_func (callable, optional): Custom metric function to use
            ignore_y (int, optional): Value to ignore in predictions and true values. Defaults to -100
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments for metric computation
        """
        super().__init__(metric_func, ignore_y, *args, **kwargs)
        self.kwargs = kwargs
        self.metrics = {"mcrmse": mcrmse}
        for key, value in metrics.__dict__.items():
            setattr(self, key, value)

    def __getattribute__(self, name):
        """
        Dynamically create regression metric computation methods.

        This method intercepts attribute access and creates wrapper functions
        for scikit-learn regression metrics, handling different input formats and
        preprocessing the data appropriately.

        Args:
            name (str): Name of the regression metric to access

        Returns:
            callable: Wrapper function for the requested regression metric
        """
        # Get the metric function
        metric_func = getattr(metrics, name, None)

        if metric_func and isinstance(metric_func, types.FunctionType):
            setattr(self, "compute", metric_func)
            # If the metric function exists, return a wrapper function

            def wrapper(y_true=None, y_score=None, *args, **kwargs):
                """
                Compute the regression metric, based on the true and predicted values.

                This wrapper handles different input formats including HuggingFace
                trainer outputs and performs necessary preprocessing for regression tasks.

                Args:
                    y_true: The true values or HuggingFace EvalPrediction object
                    y_score: The predicted values
                    ignore_y: The value to ignore in the predictions and true values in corresponding positions
                    *args: Additional positional arguments for the metric
                    **kwargs: Additional keyword arguments for the metric

                Returns:
                    dict: Dictionary containing the metric name and computed value
                """

                # This is an ugly method to handle the case when the predictions are in the form of a tuple
                # for huggingface trainers
                if y_true.__class__.__name__ == "EvalPrediction":
                    eval_prediction = y_true
                    if hasattr(eval_prediction, "label_ids"):
                        y_true = eval_prediction.label_ids
                    if hasattr(eval_prediction, "labels"):
                        y_true = eval_prediction.labels
                    predictions = eval_prediction.predictions
                    for i in range(len(predictions)):
                        if predictions[i].shape == y_true.shape and not np.all(
                            predictions[i] == y_true
                        ):
                            y_score = predictions[i]
                            break

                y_true, y_score = RegressionMetric.flatten(y_true, y_score)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_score = y_score[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))
                kwargs.update(self.kwargs)

                return {name: self.compute(y_true, y_score, *args, **kwargs)}

            return wrapper
        else:
            return super().__getattribute__(name)

    def compute(self, y_true, y_score, *args, **kwargs):
        """
        Compute the regression metric, based on the true and predicted values.

        Args:
            y_true: The true values
            y_score: The predicted values
            *args: Additional positional arguments for the metric
            **kwargs: Additional keyword arguments for the metric

        Returns:
            The computed regression metric value

        Raises:
            NotImplementedError: If no metric function is provided and compute is not implemented
        """
        if self.metric_func is not None:
            kwargs.update(self.kwargs)
            return self.metric_func(y_true, y_score, *args, **kwargs)

        else:
            raise NotImplementedError(
                "Method compute() is not implemented in the child class."
            )
