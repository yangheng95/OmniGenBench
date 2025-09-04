# -*- coding: utf-8 -*-
# file: lime_explainer.py
# time: 2025-08-28 11:19
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ...misc.utils import fprint
import numpy as np
import random
from sklearn.linear_model import Lasso
from typing import List
from ...model.classification.model import (
    OmniModelForSequenceClassification,
    OmniModelForTokenClassification,
)
from ...model.regression.model import (
    OmniModelForSequenceRegression,
    OmniModelForTokenRegression,
)
import torch


class LIMEExplainer(AbstractExplainer):
    """LIME-style Explainer for sequence models.

    This class implements a Local Interpretable Model-agnostic Explanations
    (LIME) approach adapted for biological or text sequences. It perturbs
    the input sequence, queries the model on the perturbed sequences, and
    fits a sparse linear model to identify which tokens contribute most to
    the prediction.
    """

    def __init__(self, model):
        """Initialize the LIMEExplainer.

        Args:
            model: The trained model (classification or regression) to explain.
        """
        super().__init__(model)
        self.token_to_id = {}
        self.alphabet = []
        self.num_tokens = 0

    def explain(
        self,
        sequence: str,
        task_idx: int = 0,
        batch_size: int = 32,
        num_mutations: int = 10000,
        prob_mut: float = 0.1,
        alphs=1e-5,
        seed: int = None,
        uniform: bool = False,
        **kwargs,
    ):
        """Generate LIME explanations for a given sequence.

        Args:
            sequence (str): The input sequence to explain.
            task_idx (int, optional): Target task index (for multi-task models). Defaults to 0.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            num_mutations (int, optional): Number of mutated sequences to generate. Defaults to 10000.
            prob_mut (float, optional): Probability of mutating each position. Defaults to 0.1.
            alphs (float, optional): Regularization strength for Lasso regression. Defaults to 1e-5.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            uniform (bool, optional): If True, enforce a uniform number of mutations per sequence. Defaults to False.
            **kwargs: Additional arguments passed to the model's `predict` method.

        Returns:
            np.ndarray: A coefficient matrix of shape (len(sequence), num_tokens),
                where each entry indicates the importance of a token at a given position.
        """
        fprint("Starting LIME explanation")
        # initialize the alphabet and id_to_token
        unique_chars = []
        for ch in sequence:
            if ch not in unique_chars:
                unique_chars.append(ch)
        self.token_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.alphabet = unique_chars
        self.num_tokens = len(unique_chars)

        mutated_seqs = self._generate_in_silico_mave(
            sequence, num_mutations, prob_mut, seed, uniform, **kwargs
        )
        x_muts = np.stack(
            [self._seq_to_one_hot(seq).reshape(-1) for seq in mutated_seqs]
        )
        y_muts = self._mut_predictor(mutated_seqs, task_idx, batch_size, **kwargs)

        lasso = Lasso(alpha=alphs)
        coef = lasso.fit(x_muts, y_muts).coef_
        coef = coef.reshape(len(sequence), self.num_tokens)

        return coef

    def _generate_in_silico_mave(
        self,
        sequence: str,
        num_mutations: int = 10000,
        prob_mut: float = 0.1,
        seed: int = None,
        uniform: bool = False,
        **kwargs,
    ):
        """(Private) Generates an in-silico MAVE dataset.

        This helper function creates a dataset of mutated sequences
        by randomly perturbing the input sequence at given positions.

        Args:
            sequence (str): The original input sequence.
            num_mutations (int, optional): Number of mutated sequences to generate. Defaults to 10000.
            prob_mut (float, optional): Probability of mutating each position. Defaults to 0.1.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            uniform (bool, optional): If True, enforce a fixed number of mutations per sequence. Defaults to False.
            **kwargs: Unused extra arguments.

        Returns:
            List[str]: A list of mutated sequences.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        L = len(sequence)
        avg_num_mut = int(np.ceil(L * prob_mut))
        mutated_seqs = []

        if uniform:
            num_muts = [avg_num_mut] * num_mutations
        else:
            num_muts = np.random.poisson(avg_num_mut, size=num_mutations)
            num_muts = np.clip(num_muts, 0, L)

        for n_mut in num_muts:
            seq_list = list(sequence)
            mut_positions = np.random.choice(L, n_mut, replace=False)

            for pos in mut_positions:
                original = seq_list[pos]
                alternatives = [l for l in self.alphabet if l != original]
                seq_list[pos] = random.choice(alternatives)

            mutated_seqs.append("".join(seq_list))

        return mutated_seqs

    def _seq_to_one_hot(self, sequence: str):
        """(Private) Converts a sequence to a one-hot encoding.

        This helper function converts a sequence into a one-hot encoding
        based on the provided token-to-index mapping.

        Args:
            sequence (str): The input sequence.

        Returns:
            np.ndarray: A one-hot encoded matrix of shape (len(sequence), num_tokens).
        """
        one_hot = np.zeros((len(sequence), self.num_tokens), dtype=int)
        for i, token in enumerate(sequence):
            token_id = self.token_to_id.get(token, -1)
            if token_id == -1:
                raise ValueError(f"Token {token} not found in the alphabet.")
            one_hot[i, token_id] = 1
        return one_hot

    def _mut_predictor(
        self, mut_seqs: List[str], task_idx: int = 0, batch_size: int = 32, **kwargs
    ):
        """(Private) Predicts the model's output for a list of mutated sequences.

        This helper function queries the model for the predictions of a list
        of mutated sequences.

        Args:
            mut_seqs (List[str]): List of mutated sequences.
            task_idx (int, optional): Target task index (for multi-task models). Defaults to 0.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            **kwargs: Additional arguments passed to the model's `predict` method.

        Returns:
            np.ndarray: Model predictions for each mutated sequence.
        """
        if isinstance(self.model, OmniModelForSequenceClassification) or isinstance(
            self.model, OmniModelForTokenClassification
        ):
            y_mut = []
            for i in range(0, len(mut_seqs), batch_size):
                batch = mut_seqs[i : i + batch_size]
                outputs = self.model.predict(batch, **kwargs)["logits"]
                if task_idx >= len(outputs[0]):
                    raise ValueError(
                        f"Task index {task_idx} is out of range for the model."
                    )
                y_mut.append(outputs[:, task_idx])
            y_mut = torch.cat(y_mut, dim=0).detach().cpu().numpy()
        elif isinstance(self.model, OmniModelForSequenceRegression) or isinstance(
            self.model, OmniModelForTokenRegression
        ):
            y_mut = []
            for i in range(0, len(mut_seqs), batch_size):
                batch = mut_seqs[i : i + batch_size]
                outputs = self.model.predict(batch, **kwargs)["predictions"]
                y_mut.append(outputs[:, task_idx])
            y_mut = torch.cat(y_mut, dim=0).detach().cpu().numpy()
        else:
            raise ValueError(
                f"Model type {type(self.model)} not supported for ISM explainer."
            )
        return y_mut
