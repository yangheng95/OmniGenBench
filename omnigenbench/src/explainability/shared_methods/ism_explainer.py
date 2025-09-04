# -*- coding: utf-8 -*-
# file: ism_explainer.py
# time: 2025-08-27 15:19
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ...misc.utils import fprint
import numpy as np
from ...model.classification.model import (
    OmniModelForSequenceClassification,
    OmniModelForTokenClassification,
)
from ...model.regression.model import (
    OmniModelForSequenceRegression,
    OmniModelForTokenRegression,
)
from typing import List
import torch


class ISMExplainer(AbstractExplainer):
    """Explains model predictions using the ISM method.

    ISM (In Silico MAVE) is a method that uses in-silico mutagenesis to generate a dataset, which is then
    used to train a simpler, interpretable surrogate model. From this surrogate model, additive (first-order) or pairwise (second-order) feature attributions
    can be extracted.
    """

    def __init__(self, model):
        """Initialize the ISMExplainer.

        Args:
            model: The trained model (classification or regression) to explain.
        """
        super().__init__(model)
        self.token_to_id = {}
        self.alphabet = []
        self.num_tokens = 0

    def explain(self, sequence: str, task_idx: int = 0, batch_size: int = 32, **kwargs):
        """Generate ISM explanations for a given sequence.

        Args:
            sequence (str): The input sequence to explain.
            task_idx (int, optional): Target task index (for multi-task models). Defaults to 0.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            **kwargs: Additional arguments passed to the model's `predict` method.

        Returns:
            np.ndarray: A score matrix of shape (len(sequence), num_tokens).
                - Each row corresponds to a position in the input sequence.
                - Each column corresponds to a token in the alphabet.
                - Entry [i, j] is the change in model output when the i-th token
                  is mutated to alphabet[j]. For the original base, its value is
                  set as the negative sum of the other mutations at that position.

        Raises:
            ValueError: If the task index is invalid or model type unsupported.
        """
        fprint("Starting ISM explanation")
        # initialize the alphabet and id_to_token
        unique_chars = []
        for ch in sequence:
            if ch not in unique_chars:
                unique_chars.append(ch)
        self.token_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.alphabet = unique_chars
        self.num_tokens = len(unique_chars)

        # in silico MAVE data generation
        fprint("Generating in silico MAVE data...")
        mut_seqs, mut_positions, mut_bases = self._generate_in_silico_mave(sequence)

        y_ref = self.model.predict(sequence, **kwargs)["logits"][0]
        if task_idx >= len(y_ref):
            raise ValueError(f"Task index {task_idx} is out of range for the model.")
        y_ref = y_ref[task_idx]

        # calculate the output of the model for the mutated sequences
        y_mut = self._mut_predictor(
            mut_seqs, task_idx=task_idx, batch_size=batch_size, **kwargs
        )

        scores = np.zeros(len(mut_seqs, len(self.alphabet)))

        for k, (pos, base) in enumerate(zip(mut_positions, mut_bases)):
            base_idx = self.alphabet.index(base)
            scores[k, base_idx] = y_mut[k] - y_ref

        for i in range(len(sequence)):
            orig_base = sequence[i]
            orig_idx = self.alphabet.index(orig_base)
            other_sum = np.sum(
                scores[i, j] for j in range(len(self.alphabet)) if j != orig_idx
            )
            scores[i, orig_idx] = -other_sum

        return scores

    def _generate_in_silico_mave(self, sequence: str):
        """(Private) Generate single-mutation variants of the sequence.

        This function creates all possible point mutations of the input sequence,
        by replacing each position with every other token in the alphabet.

        Args:
            sequence (str): Original input sequence.

        Returns:
            Tuple[List[str], List[int], List[str]]:
                - mut_seqs: List of mutated sequences (strings).
                - mut_positions: List of mutated positions (ints).
                - mut_bases: List of substituted bases/tokens (str).
        """
        L = len(list(sequence))

        mut_seqs = []
        mut_positions = []
        mut_bases = []

        for i in range(L):
            for j, b in enumerate(self.alphabet):
                if b == sequence[i]:
                    continue
                mut_seq = list(sequence)
                mut_seq[i] = b
                mut_seqs.append("".join(mut_seq))
                mut_positions.append(i)
                mut_bases.append(b)

        return mut_seqs, mut_positions, mut_bases

    def _mut_predictor(
        self, mut_seqs: List[str], task_idx: int = 0, batch_size: int = 32, **kwargs
    ):
        """(Private) Predict the model's output for a list of mutated sequences.

        This function uses the model to predict the output for each mutated sequence.

        Args:
            mut_seqs (List[str]): List of mutated sequences.
            task_idx (int, optional): Target task index (for multi-task models). Defaults to 0.
            batch_size (int, optional): Batch size for prediction. Defaults to 32.
            **kwargs: Additional arguments passed to the model's `predict` method.

        Returns:
            np.ndarray: Model predictions for each mutated sequence.

        Raises:
            ValueError: If the task index is invalid or model type unsupported.
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
