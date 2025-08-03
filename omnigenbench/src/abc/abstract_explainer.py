#
# Author: Shasha Zhou <sz484@exeter.ac.uk>
# Description:
#
# Copyright (C) 2020-2025. All Rights Reserved.
#
#
# Author: Shasha Zhou <sz484@exeter.ac.uk>
# Description:
#
# Copyright (C) 2020-2025. All Rights Reserved.
#
# -*- coding: utf-8 -*-
# file: abstract_explainer.py
# time: 2025-06-16 21:06
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.


from abc import ABC, abstractmethod
from typing import Any, Optional


class AbstractExplainer(ABC):
    """
    Abstract base class for all explainers.
    """

    def __init__(self, model):
        """
        Initialize the explainer.

        Args:
            model (AbstractModel): The model to explain.
            dataset (AbstractDataset): The dataset to explain, optional.
        """
        self.model = model

    @abstractmethod
    def explain(self, input: Any, **kwargs) -> Any:
        """
        Explain the model's prediction for a given input.

        Args:
            input (Any): The input to explain.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The explanation.
        """
        raise NotImplementedError(
            f"Explanation is not implemented for {self.__class__.__name__}."
        )

    def visualize(
        self, explanation: Any, inputs: Optional[Any] = None, **kwargs
    ) -> Any:
        """
        Visualize the explanation.

        Args:
            explanation (Any): The explanation to visualize.
            inputs (Any): The inputs to visualize, optional.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The visualization.
        """
        raise NotImplementedError(
            f"Visualization is not implemented for {self.__class__.__name__}."
        )

    def save(self, path: str, **kwargs) -> None:
        """
        Save the explainer.
        """
        raise NotImplementedError(
            f"Saving is not implemented for {self.__class__.__name__}."
        )

    def __call__(self, input: Any, **kwargs) -> Any:
        """
        Call the explainer.
        """
        explanation = self.explain(input, **kwargs)
        return self.visualize(explanation, input, **kwargs)
