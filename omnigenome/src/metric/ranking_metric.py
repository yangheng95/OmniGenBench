# -*- coding: utf-8 -*-
# file: ranking_metric.py
# time: 13:27 09/04/2024
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


class RankingMetric(OmniMetric):
    """
    A specialized metric class for ranking tasks and evaluation.
    
    This class provides access to ranking-specific metrics from scikit-learn
    and handles different input formats including HuggingFace trainer outputs.
    It dynamically wraps scikit-learn metrics and provides a unified interface
    for computing various ranking evaluation metrics.
    
    Attributes:
        metric_func: Custom metric function if provided
        ignore_y: Value to ignore in predictions and true values
        
    Example:
        >>> from omnigenome.src.metric import RankingMetric
        >>> metric = RankingMetric(ignore_y=-100)
        >>> y_true = [0, 1, 2, 0, 1]
        >>> y_pred = [0.1, 0.9, 0.8, 0.2, 0.7]
        >>> result = metric.roc_auc_score(y_true, y_pred)
        >>> print(result)
        {'roc_auc_score': 0.8}
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the RankingMetric class.
        
        Args:
            *args: Additional positional arguments passed to parent class
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """
        Dynamically create ranking metric computation methods.
        
        This method intercepts attribute access and creates wrapper functions
        for scikit-learn ranking metrics, handling different input formats and
        preprocessing the data appropriately.
        
        Args:
            name (str): Name of the ranking metric to access
            
        Returns:
            callable: Wrapper function for the requested ranking metric
            
        Raises:
            AttributeError: If the requested metric is not found
        """
        # Get the metric function
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            # If the metric function exists, return a wrapper function
            def wrapper(y_true=None, y_score=None, *args, **kwargs):
                """
                Compute the ranking metric, based on the true and predicted values.
                
                This wrapper handles different input formats including HuggingFace
                trainer outputs and performs necessary preprocessing for ranking tasks.
                
                Args:
                    y_true: The true values or HuggingFace EvalPrediction object
                    y_score: The predicted values (scores for ranking)
                    ignore_y: The value to ignore in the predictions and true values in corresponding positions
                    *args: Additional positional arguments for the metric
                    **kwargs: Additional keyword arguments for the metric
                    
                Returns:
                    dict: Dictionary containing the metric name and computed value
                """

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

                y_true, y_score = RankingMetric.flatten(y_true, y_score)
                y_true_mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[y_true_mask_idx]
                    try:
                        y_score = y_score[y_true_mask_idx]
                    except Exception as e:
                        warnings.warn(str(e))

                return {name: self.compute(y_true, y_score, *args, **kwargs)}

            return wrapper
        raise AttributeError(f"'CustomMetrics' object has no attribute '{name}'")

    def compute(self, y_true, y_score, *args, **kwargs):
        """
        Compute the ranking metric, based on the true and predicted values.
        
        This method should be implemented by subclasses to provide specific
        ranking metric computation logic.
        
        Args:
            y_true: The true values
            y_score: The predicted values (scores for ranking)
            *args: Additional positional arguments for the metric
            **kwargs: Additional keyword arguments for the metric
            
        Returns:
            The computed ranking metric value
            
        Raises:
            NotImplementedError: If compute method is not implemented in the child class
        """
        raise NotImplementedError(
            "Method compute() is not implemented in the child class."
        )
