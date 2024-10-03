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

from ..abc.abstract_metric import OmniGenomeMetric


class RankingMetric(OmniGenomeMetric):
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

                # This is an ugly method to handle the case when the predictions are in the form of a tuple
                # for huggingface trainers
                if y_true is not None and y_score is None:
                    if hasattr(y_true, "predictions"):
                        y_score = y_true.predictions
                    if hasattr(y_true, "label_ids"):
                        y_true = y_true.label_ids
                    if hasattr(y_true, "labels"):
                        y_true = y_true.labels
                    if len(y_score[0][1]) == np.max(y_true) + 1:
                        y_score = y_score[0]
                    else:
                        y_score = y_score[1]
                    y_score = np.argmax(y_score, axis=1)
                elif y_true is not None and y_score is not None:
                    pass  # y_true and y_score are provided
                else:
                    raise ValueError(
                        "Please provide the true and predicted values or a dictionary with 'y_true' and 'y_score'."
                    )

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
