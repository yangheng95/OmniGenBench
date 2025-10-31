# -*- coding: utf-8 -*-
# file: abstract_metric.py
# time: 12:58 09/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import numpy as np
import sklearn.metrics as metrics

from ..misc.utils import env_meta_info


class OmniMetric:
    """
    Abstract base class providing a unified interface for evaluation metrics in the OmniGenBench
    framework. This class integrates seamlessly with scikit-learn's metric ecosystem while adding
    genomics-specific functionality for handling masked labels, multi-task evaluation, and
    specialized biological metrics.

    **Design Philosophy**: This class follows the Strategy pattern, allowing interchangeable
    metric implementations while maintaining a consistent compute() interface. All scikit-learn
    metrics are automatically exposed as attributes for convenient access without explicit imports.

    **Key Features**:

    - **Scikit-learn Integration**: Automatic exposure of all sklearn.metrics functions as
      attributes (accuracy_score, f1_score, matthews_corrcoef, etc.), eliminating the need
      for separate metric imports.

    - **Masked Label Handling**: Support for PyTorch's -100 ignore convention via the ignore_y
      parameter. Labels matching ignore_y are filtered out before metric computation, essential
      for tasks with variable-length outputs or padded sequences.

    - **Flexible Computation**: The compute() method accepts various input formats (lists,
      numpy arrays, torch tensors) and returns standardized dictionary outputs for consistent
      logging and tracking.

    - **Multi-Metric Reporting**: Subclasses (ClassificationMetric, RegressionMetric,
      RankingMetric) compute multiple relevant metrics in a single call, providing comprehensive
      evaluation without manual orchestration.

    - **Custom Metric Support**: Easy extensibility through subclassing and implementing
      custom compute() methods for domain-specific metrics (e.g., Matthews Correlation
      Coefficient for imbalanced genomic datasets).

    **Common Genomic Use Cases**:

    - **Imbalanced Classification**: MCC and AUPRC for rare variant detection, where accuracy
      alone is misleading

    - **Multi-Label Prediction**: Hamming loss and F1-macro for transcription factor binding
      site prediction across hundreds of TFs

    - **Regression Tasks**: Spearman correlation for gene expression prediction, where rank
      order matters more than absolute values

    - **Token-Level Prediction**: Per-nucleotide metrics for secondary structure prediction
      and splice site detection

    **Subclass Implementations**:

    - ``ClassificationMetric``: Comprehensive classification metrics (accuracy, precision,
      recall, F1, MCC, AUROC, AUPRC) with automatic threshold selection

    - ``RegressionMetric``: Regression-specific metrics (MSE, MAE, R², Spearman/Pearson
      correlation) for continuous predictions

    - ``RankingMetric``: Ranking and retrieval metrics (NDCG, MAP, Precision@K) for
      information retrieval tasks

    Attributes:
        metric_func (callable, optional): A callable metric function from sklearn.metrics.
            If provided, used as the primary metric computation function. If None, subclasses
            should implement their own compute() method.

        ignore_y (any, optional): A value in the ground truth labels to be ignored during
            metric computation. Commonly set to -100 (PyTorch's default ignore index) or
            None. Labels matching this value are filtered out before metric calculation,
            useful for masked language modeling, padding, or variable-length sequences.

        metadata (dict): Framework metadata including version information, timestamp, and
            environment details. Automatically populated on initialization.

    Note:
        This is an abstract base class. Use task-specific subclasses for actual evaluation:

        - Use ``ClassificationMetric`` for binary/multi-class/multi-label classification
        - Use ``RegressionMetric`` for continuous value prediction
        - Use ``RankingMetric`` for ranking and retrieval tasks
        - Subclass OmniMetric for custom metrics with specialized compute() implementations

    Example:
        >>> # Access scikit-learn metrics directly
        >>> metric = OmniMetric()
        >>> acc = metric.accuracy_score(y_true, y_pred)
        >>>
        >>> # Use with ignore_y for masked tokens
        >>> metric = OmniMetric(ignore_y=-100)
        >>> # Labels of -100 will be filtered before computation
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
        Flattens the ground truth and prediction arrays. It handles various input formats and converts
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
