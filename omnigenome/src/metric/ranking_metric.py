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
    Classification metric class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        # Get the metric function
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            # If the metric function exists, return a wrapper function
            def wrapper(y_true=None, y_score=None, *args, **kwargs):
                """
                Compute the metric, based on the true and predicted values.
                :param y_true: the true values
                :param y_score: the predicted values
                :param ignore_y: the value to ignore in the predictions and true values in corresponding positions
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
        Compute the metric, based on the true and predicted values.
        :param y_true: the true values
        :param y_score: the predicted values
        """
        raise NotImplementedError(
            "Method compute() is not implemented in the child class."
        )
