# -*- coding: utf-8 -*-
# file: classification_metric.py
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


class ClassificationMetric(OmniMetric):
    """
    Classification metric class for evaluating classification models.

    This class provides a comprehensive interface for classification metrics
    in the OmniGenome framework. It integrates with scikit-learn's classification
    metrics and provides additional functionality for handling genomic classification
    tasks.

    The class automatically exposes all scikit-learn classification metrics as
    callable attributes, making them easily accessible for evaluation. It also
    handles special cases like Hugging Face's EvalPrediction objects and
    provides proper handling of ignored labels.

    Attributes:
        metric_func (callable): A callable metric function from sklearn.metrics.
        ignore_y (any): A value in the ground truth labels to be ignored during
                       metric computation. Defaults to -100.
        kwargs (dict): Additional keyword arguments for metric computation.
    """

    def __init__(self, metric_func=None, ignore_y=-100, *args, **kwargs):
        """
        Initializes the classification metric.

        Args:
            metric_func (callable, optional): A callable metric function from
                                            sklearn.metrics. If None, subclasses
                                            should implement their own compute method.
            ignore_y (any, optional): A value in the ground truth labels to be
                                    ignored during metric computation. Defaults to -100.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Example:
            >>> # Initialize with a specific metric function
            >>> metric = ClassificationMetric(metrics.accuracy_score)

            >>> # Initialize with ignore value
            >>> metric = ClassificationMetric(ignore_y=-100)
        """
        super().__init__(metric_func, ignore_y, *args, **kwargs)
        self.kwargs = kwargs

    # def __getattr__(self, name):
    def __getattribute__(self, name):
        """
        Custom attribute getter that provides dynamic access to scikit-learn metrics.

        This method provides transparent access to all scikit-learn classification
        metrics. When a metric function is accessed, it returns a callable wrapper
        that handles the metric computation with proper preprocessing.

        Args:
            name (str): The attribute name to get.

        Returns:
            callable: A wrapper function for the requested metric, or the original
                     attribute if it's not a metric function.

        Example:
            >>> metric = ClassificationMetric()
            >>> # Access any scikit-learn metric
            >>> accuracy_fn = metric.accuracy_score
            >>> result = accuracy_fn(y_true, y_pred)
        """
        # Get the metric function
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            setattr(self, "compute", metric_func)
            # If the metric function exists, return a wrapper function

            def wrapper(y_true=None, y_pred=None, *args, **kwargs):
                """
                Compute the metric, based on the true and predicted values.

                This wrapper function handles various input formats including
                Hugging Face's EvalPrediction objects and provides proper
                preprocessing for metric computation.

                Args:
                    y_true: The true values (ground truth labels).
                    y_pred: The predicted values (model predictions).
                    ignore_y: The value to ignore in the predictions and true
                             values in corresponding positions.
                    *args: Additional positional arguments for the metric function.
                    **kwargs: Additional keyword arguments for the metric function.

                Returns:
                    dict: A dictionary with the metric name as key and its value.

                Example:
                    >>> # Standard usage
                    >>> result = accuracy_fn(y_true, y_pred)
                    >>> print(result)  # {'accuracy_score': 0.85}

                    >>> # With Hugging Face EvalPrediction
                    >>> result = accuracy_fn(eval_prediction)
                    >>> print(result)  # {'accuracy_score': 0.85}
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

                y_true, y_pred = ClassificationMetric.flatten(y_true, y_pred)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_pred = y_pred[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))

                kwargs.update(self.kwargs)
                return {name: self.compute(y_true, y_pred, *args, **kwargs)}

            return wrapper
        else:
            return super().__getattribute__(name)

    def compute(self, y_true, y_pred, *args, **kwargs):
        """
        Compute the metric, based on the true and predicted values.

        This method computes the classification metric using the provided
        metric function. It handles preprocessing and applies any additional
        keyword arguments.

        Args:
            y_true: The true values (ground truth labels).
            y_pred: The predicted values (model predictions).
            *args: Additional positional arguments for the metric function.
            **kwargs: Additional keyword arguments for the metric function.

        Returns:
            dict: A dictionary with the metric name as key and its value.

        Raises:
            NotImplementedError: If no metric function is provided and the method
                              is not implemented by the subclass.

        Example:
            >>> metric = ClassificationMetric(metrics.accuracy_score)
            >>> result = metric.compute(y_true, y_pred)
            >>> print(result)  # {'accuracy_score': 0.85}
        """
        if self.metric_func is not None:
            kwargs.update(self.kwargs)
            return self.metric_func(y_true, y_pred, *args, **kwargs)
        else:
            raise NotImplementedError(
                "Method compute() is not implemented in the child class."
            )
