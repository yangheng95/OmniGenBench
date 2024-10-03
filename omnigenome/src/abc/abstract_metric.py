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


class OmniGenomeMetric:
    """
    Abstract class for all metrics, based on sklearn.metrics
    """

    def __init__(self, metric_func=None, ignore_y=None, *args, **kwargs):
        self.metric_func = metric_func
        self.ignore_y = ignore_y

        for metric in metrics.__dict__.keys():
            setattr(self, metric, metrics.__dict__[metric])

        self.metadata = env_meta_info()

    def compute(self, y_true, y_pred) -> dict:
        """
        Compute the metric, based on the true and predicted values.
        :param y_true: the true values
        :param y_pred: the predicted values
        """
        raise NotImplementedError(
            "Method compute() is not implemented in the child class. "
            "This function returns a dict containing the metric name and value."
            "e.g. {'accuracy': 0.9}"
        )

    @staticmethod
    def flatten(y_true, y_pred):
        """
        Flatten the true and predicted values.
        :param y_true: the true values
        :param y_pred: the predicted values
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        return y_true, y_pred
