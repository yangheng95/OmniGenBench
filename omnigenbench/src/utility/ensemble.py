# -*- coding: utf-8 -*-
# file: ensemble.py
# time: 21:39 24/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
from typing import List, Union

import numpy as np


class VoteEnsemblePredictor:
    """
    An ensemble predictor that combines predictions from multiple models using voting.

    This class implements ensemble methods for combining predictions from multiple
    models or checkpoints. It supports both weighted and unweighted voting, and
    provides various aggregation methods for different data types (numeric and string).

    Attributes:
        checkpoints: List of checkpoint names
        predictors: Dictionary of initialized predictors
        weights: List of weights for each predictor
        numeric_agg_func: Function for aggregating numeric predictions
        str_agg: Function for aggregating string predictions
        numeric_agg_methods: Dictionary of available numeric aggregation methods
        str_agg_methods: Dictionary of available string aggregation methods

    Example:
        >>> from omnigenbench.utility import VoteEnsemblePredictor
        >>> predictors = {
        ...     "model1": predictor1,
        ...     "model2": predictor2,
        ...     "model3": predictor3
        ... }
        >>> weights = {"model1": 1.0, "model2": 0.8, "model3": 0.6}
        >>> ensemble = VoteEnsemblePredictor(predictors, weights, numeric_agg="average")
        >>> result = ensemble.predict("ACGUAGGUAUCGUAGA")
        >>> print(result)
        {'prediction': 0.85}
    """

    def __init__(
        self,
        predictors: Union[List, dict],
        weights: Union[List, dict] = None,
        numeric_agg="average",
        str_agg="max_vote",
    ):
        """
        Initialize the VoteEnsemblePredictor.

        Args:
            predictors (List or dict): A list of checkpoints, or a dictionary of initialized predictors
            weights (List or dict, optional): A list of weights for each predictor, or a dictionary of weights for each predictor
            numeric_agg (str, optional): The aggregation method for numeric data. Options are 'average', 'mean', 'max', 'min',
                                        'median', 'mode', and 'sum'. Defaults to 'average'
            str_agg (str, optional): The aggregation method for string data. Options are 'max_vote', 'min_vote', 'vote', and 'mode'. Defaults to 'max_vote'

        Raises:
            AssertionError: If predictors and weights have different lengths or types
            AssertionError: If predictors list is empty
            AssertionError: If unsupported aggregation methods are provided
        """
        if weights is not None:
            assert len(predictors) == len(
                weights
            ), "Checkpoints and weights should have the same length"
            assert type(predictors) == type(
                weights
            ), "Checkpoints and weights should have the same type"

        assert len(predictors) > 0, "Checkpoints should not be empty"

        self.numeric_agg_methods = {
            "average": np.mean,
            "mean": np.mean,
            "max": np.max,
            "min": np.min,
            "median": np.median,
            "mode": lambda x: max(set(x), key=x.count),
            "sum": np.sum,
        }
        self.str_agg_methods = {
            "max_vote": lambda x: max(set(x), key=x.count),
            "min_vote": lambda x: min(set(x), key=x.count),
            "vote": lambda x: max(set(x), key=x.count),
            "mode": lambda x: max(set(x), key=x.count),
        }
        assert (
            numeric_agg in self.numeric_agg_methods
        ), "numeric_agg should be either: " + str(self.numeric_agg_methods.keys())
        assert (
            str_agg in self.str_agg_methods
        ), "str_agg should be either max or vote" + str(self.str_agg_methods.keys())

        self.numeric_agg_func = numeric_agg
        self.str_agg = self.str_agg_methods[str_agg]

        if isinstance(predictors, dict):
            self.checkpoints = list(predictors.keys())
            self.predictors = predictors
            self.weights = (
                list(weights.values()) if weights else [1] * len(self.checkpoints)
            )
        else:
            raise NotImplementedError(
                "Only support dict type for checkpoints and weights"
            )

    def numeric_agg(self, result: list):
        """
        Aggregate a list of numeric values.

        Args:
            result (list): A list of numeric values to aggregate

        Returns:
            The aggregated value using the specified numeric aggregation method

        Example:
            >>> ensemble = VoteEnsemblePredictor(predictors, numeric_agg="average")
            >>> result = ensemble.numeric_agg([0.8, 0.9, 0.7])
            >>> print(result)
            0.8
        """
        res = np.stack([np.array(x) for x in result])
        return self.numeric_agg_methods[self.numeric_agg_func](res, axis=0)

    def __ensemble(self, result: dict):
        """
        Aggregate prediction results by calling the appropriate aggregation method.

        This method determines the type of result and calls the appropriate
        aggregation method (numeric or string).

        Args:
            result (dict): A dictionary containing the prediction results

        Returns:
            The aggregated prediction result
        """
        if isinstance(result, dict):
            return self.__dict_aggregate(result)
        elif isinstance(result, list):
            return self.__list_aggregate(result)
        else:
            return result

    def __dict_aggregate(self, result: dict):
        """
        Recursively aggregate a dictionary of prediction results.

        This method recursively processes nested dictionaries and applies
        appropriate aggregation methods to each level.

        Args:
            result (dict): A dictionary containing the prediction results

        Returns:
            dict: The aggregated prediction result
        """
        ensemble_result = {}
        for k, v in result.items():
            if isinstance(result[k], list):
                ensemble_result[k] = self.__list_aggregate(result[k])
            elif isinstance(result[k], dict):
                ensemble_result[k] = self.__dict_aggregate(result[k])
            else:
                ensemble_result[k] = result[k]
        return ensemble_result

    def __list_aggregate(self, result: list):
        """
        Aggregate a list of prediction results.

        This method handles different types of list elements and applies
        appropriate aggregation methods based on the data type.

        Args:
            result (list): A list of prediction results to aggregate

        Returns:
            The aggregated result

        Raises:
            AssertionError: If all elements in the list are not of the same type
        """
        if not isinstance(result, list):
            result = [result]

        assert all(
            isinstance(x, (type(result[0]))) for x in result
        ), "all type of result should be the same"

        if isinstance(result[0], list):
            for i, k in enumerate(result):
                result[i] = self.__list_aggregate(k)
            # start to aggregate
            try:
                new_result = self.numeric_agg(result)
            except Exception as e:
                try:
                    new_result = self.str_agg(result)
                except Exception as e:
                    new_result = result
            return [new_result]

        elif isinstance(result[0], dict):
            for k in result:
                result[k] = self.__dict_aggregate(result[k])
            return result

        # start to aggregate
        try:
            new_result = self.numeric_agg(result)
        except Exception as e:
            try:
                new_result = self.str_agg(result)
            except Exception as e:
                new_result = result

        return new_result

    def predict(self, text, ignore_error=False, print_result=False):
        """
        Predicts on a single text and returns the ensemble result.

        This method combines predictions from all predictors in the ensemble
        using the specified weights and aggregation methods.

        Args:
            text (str): The text to perform prediction on
            ignore_error (bool, optional): Whether to ignore any errors that occur during prediction. Defaults to False
            print_result (bool, optional): Whether to print the prediction result. Defaults to False

        Returns:
            dict: The ensemble prediction result

        Example:
            >>> result = ensemble.predict("ACGUAGGUAUCGUAGA", ignore_error=True)
            >>> print(result)
            {'prediction': 0.85, 'confidence': 0.92}
        """
        # Initialize an empty dictionary to store the prediction result
        result = {}
        # Loop through each checkpoint and predictor in the ensemble
        for ckpt, predictor in self.predictors.items():
            # Perform prediction on the text using the predictor
            raw_result = predictor.inference(
                text, ignore_error=ignore_error, print_result=print_result
            )
            # For each key-value pair in the raw result dictionary
            for key, value in raw_result.items():
                # If the key is not already in the result dictionary
                if key not in result:
                    # Initialize an empty list for the key
                    result[key] = []
                # Append the value to the list the number of times specified by the corresponding weight
                for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                    result[key].append(value)
        # Return the ensemble result by aggregating the values in the result dictionary
        return self.__ensemble(result)

    def batch_predict(self, texts, ignore_error=False, print_result=False):
        """
        Predicts on a batch of texts using the ensemble of predictors.

        This method processes multiple texts efficiently by combining predictions
        from all predictors in the ensemble for each text in the batch.

        Args:
            texts (list): A list of strings to predict on
            ignore_error (bool, optional): Boolean indicating whether to ignore errors or raise exceptions when prediction fails. Defaults to False
            print_result (bool, optional): Boolean indicating whether to print the raw results for each predictor. Defaults to False

        Returns:
            list: A list of dictionaries, each dictionary containing the aggregated results of the corresponding text in the input list

        Example:
            >>> texts = ["ACGUAGGUAUCGUAGA", "GGCTAGCTA", "TATCGCTA"]
            >>> results = ensemble.batch_predict(texts, ignore_error=True)
            >>> print(len(results))
            3
        """
        batch_raw_results = []
        for ckpt, predictor in self.predictors.items():
            if hasattr(predictor, "inference"):
                raw_results = predictor.inference(
                    texts,
                    ignore_error=ignore_error,
                    print_result=print_result,
                    merge_results=False,
                )
            else:
                raw_results = predictor.inference(
                    texts, ignore_error=ignore_error, print_result=print_result
                )
            batch_raw_results.append(raw_results)

        batch_results = []
        for raw_result in batch_raw_results:
            for i, result in enumerate(raw_result):
                if i >= len(batch_results):
                    batch_results.append({})
                for key, value in result.items():
                    if key not in batch_results[i]:
                        batch_results[i][key] = []
                    for _ in range(self.weights[self.checkpoints.index(ckpt)]):
                        batch_results[i][key].append(value)

        ensemble_results = []
        for result in batch_results:
            ensemble_results.append(self.__ensemble(result))
        return ensemble_results

    # def batch_predict(self, texts, ignore_error=False, print_result=False):
    #     batch_results = []
    #     for text in tqdm.tqdm(texts, desc='Batch predict: '):
    #         result = self.predict(text, ignore_error=ignore_error, print_result=print_result)
    #         batch_results.append(result)
    #     return batch_results
