# -*- coding: utf-8 -*-
# file: classification_metric.py
# time: 12:57 09/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.

import types
import numpy as np
import sklearn.metrics as metrics

from ..abc.abstract_metric import OmniGenomeMetric


class ClassificationMetric(OmniGenomeMetric):
    """
    Classification metric class
    """

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        super().__init__(metric_func, ignore_y, *args, **kwargs)
        self.kwargs = kwargs

    # def __getattr__(self, name):
    def __getattribute__(self, name):
        # Get the metric function
        metric_func = getattr(metrics, name, None)
        if metric_func and isinstance(metric_func, types.FunctionType):
            setattr(self, "compute", metric_func)
            # If the metric function exists, return a wrapper function

            def wrapper(y_true, y_pred, *args, **kwargs):
                """
                Compute the metric, based on the true and predicted values.
                :param y_true: the true values
                :param y_pred: the predicted values
                :param ignore_y: the value to ignore in the predictions and true values in corresponding positions
                """
                y_true, y_pred = ClassificationMetric.flatten(y_true, y_pred)
                mask_idx = np.where(y_true != self.ignore_y)
                if self.ignore_y is not None:
                    y_true = y_true[mask_idx]
                    y_pred = y_pred[mask_idx]

                kwargs.update(self.kwargs)
                return {name: self.compute(y_true, y_pred, *args, **kwargs)}

            return wrapper
        else:
            return super().__getattribute__(name)

    def compute(self, y_true, y_pred, *args, **kwargs):
        """
        Compute the metric, based on the true and predicted values.
        :param y_true: the true values
        :param y_pred: the predicted values
        :param ignore_y: the value to ignore in the predictions and true values in corresponding positions
        """
        if self.metric_func is not None:
            kwargs.update(self.kwargs)
            return self.metric_func(y_true, y_pred, *args, **kwargs)
        else:
            raise NotImplementedError(
                "Method compute() is not implemented in the child class."
            )
