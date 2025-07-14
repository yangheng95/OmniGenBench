# -*- coding: utf-8 -*-
# file: abstract_metric.py
# time: 12:58 09/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import numpy as np
import sklearn.metrics as metrics

from ..misc.utils import env_meta_info


class OmniMetric:
    """
    Abstract base class for all metrics in OmniGenome, based on scikit-learn.

    This class provides a unified interface for evaluation metrics in the OmniGenome
    framework. It integrates with scikit-learn's metric functions and provides
    additional functionality for handling genomic data evaluation.

    The class automatically exposes all scikit-learn metrics as attributes,
    making them easily accessible for evaluation tasks.

    Attributes:
        metric_func (callable): A callable metric function from `sklearn.metrics`.
        ignore_y (any): A value in the ground truth labels to be ignored during
                       metric computation.
        metadata (dict): Metadata about the metric including version info.
    """

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        """
        Initializes the metric.

        Args:
            metric_func (callable, optional): A callable metric function from
                                            `sklearn.metrics`. If None, subclasses
                                            should implement their own compute method.
            ignore_y (any, optional): A value in the ground truth labels to be
                                    ignored during metric computation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Example:
            >>> # Initialize with a specific metric function
            >>> metric = OmniMetric(metrics.accuracy_score)

            >>> # Initialize with ignore value
            >>> metric = OmniMetric(ignore_y=-100)
        """
        self.metric_func = metric_func
        self.ignore_y = ignore_y

        # Expose all scikit-learn metrics as attributes
        for metric in metrics.__dict__.keys():
            setattr(self, metric, metrics.__dict__[metric])

        self.metadata = env_meta_info()

    def compute(self, y_true, y_pred) -> dict:
        """
        Computes the metric. This method must be implemented by subclasses.

        This method should be implemented by concrete metric classes to define
        how the metric is calculated for their specific evaluation task.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            dict: A dictionary with the metric name as key and its value.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        Example:
            >>> # In a classification metric
            >>> result = metric.compute(y_true, y_pred)
            >>> print(result)  # {'accuracy': 0.85}
        """
        raise NotImplementedError(
            "Method compute() is not implemented in the child class. "
            "This function returns a dict containing the metric name and value."
            "e.g. {'accuracy': 0.9}"
        )

    @staticmethod
    def flatten(y_true, y_pred):
        """
        Flattens the ground truth and prediction arrays.

        This utility method ensures that the input arrays are properly flattened
        for metric computation. It handles various input formats and converts
        them to 1D numpy arrays.

        Args:
            y_true: Ground truth labels in any format that can be converted to numpy array.
            y_pred: Predicted labels in any format that can be converted to numpy array.

        Returns:
            tuple: A tuple of flattened `y_true` and `y_pred` as numpy arrays.

        Example:
            >>> y_true = [[1, 2], [3, 4]]
            >>> y_pred = [[1, 2], [3, 4]]
            >>> flat_true, flat_pred = OmniMetric.flatten(y_true, y_pred)
            >>> print(flat_true.shape)  # (4,)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return y_true, y_pred
