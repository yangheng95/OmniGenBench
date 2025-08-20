# -*- coding: utf-8 -*-
# file: sqyud_explainer.py
# time: 2025-06-23 15:19
# author: Shasha Zhou <sz484@exeter.ac.uk>
# Copyright (C) 2020-2025. All Rights Reserved.

from ...abc.abstract_explainer import AbstractExplainer
from ...misc.utils import fprint
from typing import List, Optional, Tuple, Dict
import numpy as np
import random
from itertools import combinations, product
from ...model.classification.model import (
    OmniModelForSequenceClassification,
    OmniModelForTokenClassification,
)
from ...model.regression.model import (
    OmniModelForSequenceRegression,
    OmniModelForTokenRegression,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import math
import os


class SQUIDExplainer(AbstractExplainer):
    """Explains model predictions using the SQUID method.

    SQUID (Surrogate-based QUantitative-epistatic-Interaction-Discovery) is a
    method that uses in-silico mutagenesis to generate a dataset, which is then
    used to train a simpler, interpretable surrogate model. From this surrogate
    model, additive (first-order) or pairwise (second-order) feature attributions
    can be extracted.

    Attributes:
        model: The target deep learning model to explain.
        gpmap (str): The type of genotype-phenotype map for the surrogate model,
                     either 'additive' or 'pairwise'.
        token_to_id (Dict[str, int]): A mapping from sequence characters to integer IDs.
        alphabet (List[str]): The list of unique characters in the input sequence.
        num_tokens (int): The size of the alphabet.

    Reference:
        Seitz, E.E., McCandlish, D.M., Kinney, J.B., and Koo P.K. Interpreting
        cis-regulatory mechanisms from genomic deep neural networks using surrogate
        models. Nat Mach Intell (2024).
        https://doi.org/10.1038/s42256-024-00851-5
    """

    def __init__(self, model, gpmap: str = "additive", **kwargs):
        """Initializes the SQUIDExplainer.

        Args:
            model: The pre-trained model to be explained.
            gpmap (str, optional): The type of surrogate model to fit. Can be 'additive'
                                   for first-order effects or 'pairwise' for second-order
                                   effects. Defaults to "additive".
            **kwargs: Additional keyword arguments.
        """
        super().__init__(model)
        self.model = model
        self.token_to_id = {}
        self.alphabet = []
        self.num_tokens = 0
        self.gpmap = gpmap

    def explain(
        self,
        sequence: str,
        mut_type: str = "random",
        mut_rate: float = 0.1,
        uniform: bool = False,
        max_order: int = -1,
        mut_window: Optional[Tuple[int, int]] = None,
        inter_window: Optional[Tuple[int, int]] = None,
        context_agnositc: bool = False,
        num_sim: int = 10000,
        seed: Optional[int] = None,
        save_window: Optional[Tuple[int, int]] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        """
        Generates feature attributions for an input sequence using the SQUID method.

        This method performs three main steps:
            Generates a dataset of mutated sequences and their corresponding model
            predictions (in-silico MAVE).
            Trains an interpretable surrogate model on this dataset.
            Extracts the learned parameters from the surrogate model, which represent
            the feature attributions.

        Args:
            sequence (str): The input sequence to explain.
            mut_type (str, optional): The mutagenesis strategy. Can be "random" or
                                      "combinatorial". Defaults to "random".
            mut_rate (float, optional): The average mutation rate for 'random' mutagenesis.
                                        Defaults to 0.1.
            uniform (bool, optional): If True, use a fixed number of mutations per sequence
                                      for 'random' mutagenesis. Defaults to False.
            max_order (int, optional): The maximum order of mutations for 'combinatorial'
                                       mutagenesis. -1 means all orders. Defaults to -1.
            mut_window (Tuple[int, int], optional): The (start, end) window within the
                                                    sequence to apply mutations. Defaults to None.
            inter_window (Tuple[int, int], optional): A window for inter-mutational analysis.
                                                      Defaults to None.
            context_agnositc (bool, optional): If True, randomize the context outside the
                                               mutation window. Defaults to False.
            num_sim (int, optional): The number of mutated sequences to generate.
                                     Defaults to 10000.
            seed (Optional[int], optional): A random seed for reproducibility. Defaults to None.
            save_window (Tuple[int, int], optional): The window of the sequence to use for
                                                     training the surrogate model. Defaults to None.
            batch_size (int, optional): Batch size for getting predictions from the target model.
                                        Defaults to 32.
            **kwargs: Additional arguments passed to the surrogate model fitting process.

        Returns:
            np.ndarray: The learned parameters from the surrogate model.
                        - If `gpmap` is 'additive', returns `theta_lc` with shape (L, A),
                          representing first-order effects.
                        - If `gpmap` is 'pairwise', returns `theta_lclc` with shape (L, A, L, A),
                          representing second-order effects.
        """
        fprint("Starting SQUID explanation")
        print(f"Using gpmap type: {self.gpmap}")
        # initialize the alphabet and id_to_token
        unique_chars = []
        for ch in sequence:
            if ch not in unique_chars:
                unique_chars.append(ch)
        self.token_to_id = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.alphabet = unique_chars
        self.num_tokens = len(unique_chars)

        # in silico MAVE data generation
        print("Generating in silico MAVE data...")
        full_seqs, x_mut, y_mut = self._generate_in_silico_mave(
            sequence,
            mut_type,
            mut_rate,
            uniform,
            max_order,
            mut_window,
            inter_window,
            context_agnositc,
            num_sim,
            seed,
            save_window,
            batch_size,
            **kwargs,
        )

        # build and train surrogate model
        print("Building and training surrogate model...")
        surrogate_model = SQUIDSurrogateModel(
            input_shape=x_mut.shape,
            gpmap=self.gpmap,
            num_tasks=len(y_mut[0]),
            token_to_id=self.token_to_id,
            seed=seed,
        )
        surrogate_model.fit(x_mut, y_mut)
        theta_0, theta_lc, theta_lclc = surrogate_model.get_params()
        if self.gpmap == "additive":
            return theta_lc
        elif self.gpmap == "pairwise":
            return theta_lclc

    def _generate_in_silico_mave(
        self,
        sequence: str,
        mut_type: str = "random",
        mut_rate: float = 0.1,
        uniform: bool = False,
        max_order: int = -1,
        mut_window: Optional[Tuple[int, int]] = None,
        inter_window: Optional[Tuple[int, int]] = None,
        context_agnositc: bool = False,
        num_sim: int = 10000,
        seed: Optional[int] = None,
        save_window: Optional[Tuple[int, int]] = None,
        batch_size: int = 32,
        **kwargs,
    ):
        """(Private) Generates an in-silico MAVE dataset.

        This helper function creates a dataset of mutated sequences and uses the
        provided deep learning model to predict their corresponding phenotypes.

        Args:
            (See `explain` method for argument descriptions)

        Returns:
            Tuple[List[str], np.ndarray, np.ndarray]: A tuple containing:
                - A list of the generated full sequences (strings).
                - The one-hot encoded mutated sequences (`x_mut`).
                - The model predictions for each sequence (`y_mut`).
        """

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if mut_type == "random":
            mutagenesis = SQUIDRandomMutagenesis(
                alphabet=self.alphabet, mut_rate=mut_rate, uniform=uniform, seed=seed
            )
        elif mut_type == "combinatorial":
            mutagenesis = SQUIDCombinatorialMutagenesis(
                alphabet=self.alphabet,
                max_order=max_order,
                mut_window=mut_window,
                seed=seed,
            )
        else:
            raise ValueError(
                f"Invalid mut_type: {mut_type}. Must be 'random' or 'combinatorial'."
            )

        L = len(sequence)
        full_seqs = []

        # generate in silico MAVE based on mutagenesis strategy
        if mut_window is not None:
            start, end = mut_window
            assert (
                start >= 0 and end <= L
            ), f"Invalid mut_window: {mut_window}. Must be within the sequence length [0, {L}]."
            target_seqs = sequence[start:end]
            mutated_parts = mutagenesis(target_seqs, num_sim, **kwargs)
        else:
            mutated_parts = mutagenesis(sequence, num_sim, **kwargs)
            x_mut = [self._one_hot_encode(m_seq) for m_seq in mutated_parts]
            x_mut = np.stack(x_mut)
            y_mut = self._mut_predictor(mutated_parts, batch_size, **kwargs)
            y_mut = np.array(y_mut)
            return (
                mutated_parts,
                x_mut,
                y_mut,
            )  # No mutated window, return all mutated parts

        for m_seq in mutated_parts:
            # Add context
            if context_agnositc:
                left = "".join(random.choice(self.alphabet, k=start))
                right = "".join(random.choice(self.alphabet, k=L - end))
            else:
                left = sequence[:start]
                right = sequence[end:]
            full_seq = left + m_seq + right

            # Add inter-mutated region
            if inter_window is not None:
                inters = (
                    inter_window
                    if isinstance(inter_window[0], list)
                    else [inter_window]
                )
                for w_start, w_end in inters:
                    assert (
                        w_start >= 0 and w_end <= L
                    ), f"Invalid inter_window: {inter_window}. Must be within the sequence length [0, {L}]."
                    inter_seq = "".join(random.choice(self.alphabet, k=w_end - w_start))
                    full_seq = full_seq[:w_start] + inter_seq + full_seq[w_end:]

            full_seqs.append(full_seq)

        if save_window is not None:
            start, end = save_window
            assert (
                start >= 0 and end <= L
            ), f"Invalid save_window: {save_window}. Must be within the sequence length [0, {L}]."
            if mut_window is not None:
                if save_window[0] > mut_window[0] or save_window[1] < mut_window[1]:
                    start = 0
                    end = L
                    fprint(
                        "Conflicting save_window and mut_window. save_window is ignored."
                    )
        else:
            start, end = 0, L
        x_mut = [self._one_hot_encode(full_seq[start:end]) for full_seq in full_seqs]
        x_mut = np.stack(x_mut)
        y_mut = self._mut_predictor(full_seqs, batch_size, **kwargs)
        y_mut = np.array(y_mut)
        return full_seqs, x_mut, y_mut

    def _mut_predictor(self, mutated_parts: List[str], batch_size: int = 32, **kwargs):
        """
        Predict the output of the model for the mutated parts.

        Args:
            sequence: the sequence to explain, string
            mutated_parts: the mutated parts, list of strings
            **kwargs: additional arguments

        Returns:
            y_mut: the output of the model for the mutated parts, numpy array
        """

        if isinstance(self.model, OmniModelForSequenceClassification) or isinstance(
            self.model, OmniModelForTokenClassification
        ):
            y_mut = []
            for i in range(0, len(mutated_parts), batch_size):
                batch = mutated_parts[i : i + batch_size]
                y_mut.append(self.model.predict(batch, **kwargs)["logits"])
            y_mut = torch.cat(y_mut, dim=0).detach().cpu().numpy()
        elif isinstance(self.model, OmniModelForSequenceRegression) or isinstance(
            self.model, OmniModelForTokenRegression
        ):
            y_mut = []
            for i in range(0, len(mutated_parts), batch_size):
                batch = mutated_parts[i : i + batch_size]
                y_mut.append(self.model.predict(batch, **kwargs)["predictions"])
            y_mut = torch.cat(y_mut, dim=0).detach().cpu().numpy()
        else:
            raise ValueError(
                f"Model type {type(self.model)} not supported for SQUID explainer."
            )
        return y_mut

    def _one_hot_encode(self, sequence: str):
        """
        One-hot encode the sequence.

        Args:
            sequence: the sequence to encode, string

        Returns:
            one_hot: the one-hot encoded sequence, numpy array
        """
        one_hot = np.zeros((len(sequence), self.num_tokens), dtype=int)
        for i, token in enumerate(sequence):
            token_id = self.token_to_id.get(token, -1)
            if token_id == -1:
                raise ValueError(f"Token {token} not found in the alphabet.")
            one_hot[i, token_id] = 1
        return one_hot

    # def _build_surrogate_model(self, X_mut, y_mut):
    #     """
    #     Build a surrogate model for the mutated parts.
    #     """


class SQUIDBaseMutagenesis:
    """
    SQUIDBaseMutagenesis is a class that generates in silico MAVE data for a given sequence.
    """

    def __call__(self, sequence: str, num_sim: int = 100, **kwargs):
        """
        Return an in silico MAVE based on the given sequence.

        Args:
            sequence: the sequence to mutate, string
            num_sim: the number of simulations to generate, default is 100, int
            **kwargs: additional arguments

        Returns:
            list: a list of mutated sequences
        """
        raise NotImplementedError("This method should be implemented by the subclass.")


class SQUIDRandomMutagenesis(SQUIDBaseMutagenesis):
    """
    SQUIDRandomMutagenesis is a class that generates in silico MAVE data for a given sequence using random mutagenesis.
    """

    def __init__(
        self,
        alphabet: List[str],
        mut_rate: float = 0.1,
        uniform: bool = False,
        seed: Optional[int] = None,
    ):
        self.mut_rate = mut_rate
        self.uniform = uniform
        self.seed = seed
        self.alphabet = alphabet

    def __call__(self, sequence: str, num_sim: int = 10000, **kwargs):
        """
        Return an in silico MAVE based on the given sequence.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        L = len(sequence)
        avg_num_mut = int(np.ceil(L * self.mut_rate))
        mutated_seqs = []

        if self.uniform:
            num_muts = [avg_num_mut] * num_sim
        else:
            num_muts = np.random.poisson(avg_num_mut, size=num_sim)
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


class SQUIDCombinatorialMutagenesis(SQUIDBaseMutagenesis):
    """
    SQUIDCombinatorialMutagenesis is a class that generates in silico MAVE data for a given sequence using combinatorial mutagenesis.
    """

    def __init__(
        self,
        alphabet: List[str],
        max_order: int = 1,
        mut_window: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the SQUIDCombinatorialMutagenesis class.

        Args:
            max_order: the maximum order of mutations to generate. If -1, generates all possible mutations. If 1, generates only single mutations (all SNVs). If 2, generates single and double mutations, etc. int, default is 1

            mut_window: [int, int], optional. Index of start and end of the mutated region. If provided, only generates mutations within this window (inclusive on both ends). For example, mut_window=(4, 6) will generate mutations at positions 4, 5, and 6. (Defaults to None, which means the entire sequence is mutated)

            seed: the seed for the random number generator, default is None, int
        """
        self.max_order = max_order
        self.mut_window = mut_window
        self.seed = seed
        self.alphabet = alphabet

    def __call__(self, sequence: str, num_sim: int = 10000, **kwargs):
        """
        Return an in silico MAVE based on the given sequence.

        Args:
            sequence: the sequence to mutate, string
            num_sim: this parameter is ignored, int
            **kwargs: additional arguments

        Returns:
            list: a list of mutated sequences
        """
        if self.max_order < -1:
            raise ValueError(
                f"Invalid max_order: {self.max_order}. Must be -1 or greater."
            )

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        L = len(sequence)

        if self.mut_window is not None:
            start_pos, end_pos = self.mut_window
            end_pos += 1
            if start_pos < 0 or end_pos > L:
                raise ValueError(
                    f"Invalid mut_window: {self.mut_window}. Must be within the sequence length [0, {L}]."
                )
        else:
            start_pos, end_pos = 0, L

        window_len = end_pos - start_pos
        if self.max_order > window_len:
            raise ValueError(
                f"max_order: {self.max_order} is greater than the window length: {window_len}."
            )

        max_order = window_len if self.max_order == -1 else self.max_order

        ref_seq = list(sequence)
        mutated_seqs = ["".join(ref_seq)]

        # generate all possible alternative bases for each position
        alt_base_dict = {
            i: [l for l in self.alphabet if l != ref_seq[i]]
            for i in range(start_pos, end_pos)
        }

        # generate all possible mutations
        for order in range(1, max_order + 1):
            mut_pos_combinations = list(combinations(range(start_pos, end_pos), order))
            total_variants = len(mut_pos_combinations) * (3**order)

            for positions in mut_pos_combinations:
                alt_base_lists = [alt_base_dict[pos] for pos in positions]
                for alt_bases in product(*alt_base_lists):
                    new_seq = ref_seq.copy()
                    for pos, alt_base in zip(positions, alt_bases):
                        new_seq[pos] = alt_base
                    mutated_seqs.append("".join(new_seq))

        return mutated_seqs


class SQUIDAdditiveGPMap(nn.Module):
    """Additive genotype‑phenotype map: φ = θ_0 + Σ_{l,c} θ_{l,c} x_{l,c}."""

    def __init__(self, L: int, A: int, reg_strength: float = 0.0):
        super().__init__()
        # θ_{l,c}. Shape (L, A)
        self.theta_lc = nn.Parameter(torch.zeros(L, A))
        # θ_0 – scalar bias.
        self.theta_0 = nn.Parameter(torch.zeros(1))
        self.reg_strength = reg_strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is one‑hot with shape (N, L, A). Outputs latent φ with shape (N, 1)."""
        # (N, L, A) · (L, A) → (N, L, A) → sum_{l,c}
        phi = (x * self.theta_lc).sum(dim=(1, 2)) + self.theta_0  # (N,)
        return phi.unsqueeze(-1)  # (N,1)

    def l2_regularizer(self) -> torch.Tensor:
        if self.reg_strength == 0:
            return torch.tensor(0.0, device=self.theta_lc.device)
        return self.reg_strength * (self.theta_lc.pow(2).mean())


class SQUIDPairwiseGPMap(nn.Module):
    """Full pairwise model: φ = θ_0 + Σ θ_{l,c} x_{l,c} + Σ θ_{l1,c1,l2,c2} x_{l1,c1} x_{l2,c2}.
    The interaction tensor is stored in a factorised low‑rank form so we can
    scale to reasonable sequence lengths without a massive O(L²A²) memory
    footprint.  We use a CP‑decomposition with *K* latent factors."""

    def __init__(self, L: int, A: int, rank: int = 8, reg_strength: float = 0.0):
        super().__init__()
        self.L, self.A = L, A
        self.rank = rank
        self.theta_lc = nn.Parameter(torch.zeros(L, A))
        self.theta_0 = nn.Parameter(torch.zeros(1))
        # low‑rank factors: for each position‑char we have a rank‑dim embedding
        self.U = nn.Parameter(torch.randn(L, A, rank) * 0.01)
        self.reg_strength = reg_strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # additive part
        additive = (x * self.theta_lc).sum(dim=(1, 2))  # (N,)
        # pairwise part via CP‑factorised tensor
        #   φ_pair = Σ_{k} (Σ_{l,c} U_{l,c,k} x_{l,c})² − Σ_{l,c} (U_{l,c,k} x_{l,c})²
        # The subtraction removes the diagonal terms.
        # Compute projections along each rank component
        proj = torch.einsum("nlc,lck->nk", x, self.U)  # (N, rank)
        pairwise = 0.5 * (
            proj.pow(2).sum(dim=1)
            - (x.pow(2) * self.U.pow(2).sum(dim=-1)).sum(dim=(1, 2))
        )
        phi = additive + pairwise + self.theta_0  # (N,)
        return phi.unsqueeze(-1)

    def l2_regularizer(self) -> torch.Tensor:
        if self.reg_strength == 0:
            return torch.tensor(0.0, device=self.theta_lc.device)
        reg = self.theta_lc.pow(2).mean() + self.U.pow(2).mean()
        return self.reg_strength * reg


class SQUIDGlobalEpistasis(nn.Module):
    """Simple 1‑hidden‑layer sigmoid‑basis network to model GE non‑linearity."""

    def __init__(self, hidden_nodes: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_nodes), nn.Sigmoid(), nn.Linear(hidden_nodes, 1)
        )

    def forward(self, z):
        return self.net(z)


class SQUIDSurrogateModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_tasks: int,
        gpmap: str = "additive",
        regression_type: str = "GE",
        linearity: str = "nonlinear",
        noise: str = "Gaussian",
        noise_order: int = 0,
        reg_strength: float = 0.1,
        hidden_nodes: int = 50,
        token_to_id: Dict[str, int] = {},
        deduplicate: bool = True,
        gpu: bool = True,
        pairwise_rank: int = 8,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.N, self.L, self.A = input_shape
        self.gpmap_type = gpmap
        self.regression_type = regression_type
        self.linearity = linearity
        self.noise = noise
        self.noise_order = noise_order if linearity == "nonlinear" else 0
        self.reg_strength = reg_strength
        self.hidden_nodes = hidden_nodes
        self.token_to_id = (
            token_to_id if token_to_id is not None else {"A": 0, "U": 1, "G": 2, "C": 3}
        )
        self.deduplicate = deduplicate
        self.device = torch.device(
            "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
        )
        self.seed = seed

        # Build GP‑map
        if self.gpmap_type == "additive":
            self.gpmap = SQUIDAdditiveGPMap(self.L, self.A, reg_strength=reg_strength)
        elif self.gpmap_type == "pairwise":
            print(f"Using pairwise gpmap with rank: {pairwise_rank}")
            self.gpmap = SQUIDPairwiseGPMap(
                self.L, self.A, rank=pairwise_rank, reg_strength=reg_strength
            )
        else:
            raise ValueError(f"Unsupported gpmap type: {self.gpmap_type}")

        # Non‑linearity (for GE regressions)
        if self.linearity == "nonlinear":
            self.nonlinearity = SQUIDGlobalEpistasis(hidden_nodes=hidden_nodes)
        else:
            self.nonlinearity = nn.Identity()

        # Final task‑specific heads
        if self.regression_type == "GE":
            # Continuous output; one regression head per task
            self.head = nn.Linear(1, num_tasks)
        elif self.regression_type == "MPA":
            # Classification; assume y ∈ {0,…,C‑1}. We model logits per class.
            self.head = nn.Linear(1, num_tasks)
        else:
            raise ValueError("regression_type must be 'GE' or 'MPA'")

        self.to(self.device)

    def dataframe(
        self, x: np.ndarray, y: np.ndarray
    ) -> "Tuple[List[str], torch.Tensor]":
        """MAVE‑NN wanted a pandas DataFrame.  Here we simply return *seq_list*
        (for inspection) and a PyTorch tensor (N,) or (N,num_tasks) for y.
        """

        def one_hot_to_indices(x: torch.Tensor) -> torch.Tensor:
            """Convert one‑hot encoded sequences to indices."""
            return torch.argmax(x, dim=-1)

        def indices_to_seqs(
            indices: torch.Tensor, token_to_id: Dict[str, int]
        ) -> List[str]:
            """Convert indices to sequences."""
            return [
                "".join([list(token_to_id.keys())[i] for i in indice])
                for indice in indices
            ]

        seq_ids = one_hot_to_indices(torch.tensor(x, dtype=torch.long))
        seq_list = indices_to_seqs(seq_ids, self.token_to_id)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return seq_list, y_tensor

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 5e-4,
        epochs: int = 500,
        batch_size: int = 128,
        early_stopping: bool = True,
        patience: int = 25,
        save_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> Tuple[nn.Module, List[str]]:
        """End‑to‑end training loop with a 60/20/20 train/val/test split.
        Returns the trained *nn.Module* and the sequence list (for external
        evaluation if needed)."""

        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)  # (N,L,A)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Deduplicate if requested
        if self.deduplicate:
            unique, indices = np.unique(
                np.concatenate([x.reshape(self.N, -1), y.reshape(self.N, -1)], axis=1),
                axis=0,
                return_index=True,
            )
            x_tensor = x_tensor[indices]
            y_tensor = y_tensor[indices]

        # Build dataset & split
        dataset = TensorDataset(x_tensor, y_tensor)
        n_total = len(dataset)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        n_test = n_total - n_train - n_val
        self.train_set, self.val_set, self.test_set = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(
                self.seed if self.seed is not None else 42
            ),
        )

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size)

        # Optimiser
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        best_val_loss = math.inf
        epochs_without_improve = 0
        self.train_history = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = self._nll(xb, yb)
                (loss + self.gpmap.l2_regularizer()).backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / n_train

            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = (
                    sum(
                        self._nll(xb.to(self.device), yb.to(self.device)).item()
                        * xb.size(0)
                        for xb, yb in val_loader
                    )
                    / n_val
                )

            self.train_history["loss"].append(avg_loss)
            self.train_history["val_loss"].append(val_loss)

            if verbose and epoch % 10 == 0:
                print(
                    f"Epoch {epoch:4d} | loss={avg_loss:.4f} | val_loss={val_loss:.4f}"
                )

            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss - 1e-4:  # significant improvement
                    best_val_loss = val_loss
                    epochs_without_improve = 0
                    best_state = self.state_dict()
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        if verbose:
                            print("Early stopping triggered.")
                        break

        # Restore best weights if early stopping
        if early_stopping:
            self.load_state_dict(best_state)

        # Optionally save
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                self.state_dict(), os.path.join(save_dir, "surrogate_mavenn_pt.pt")
            )

        seq_list, _ = self.dataframe(x, y)
        return self, seq_list

    def _nll(self, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """Negative log‑likelihood / loss function depending on regression type
        and noise model."""
        preds = self._predict_continuous(xb)  # (N,num_tasks) continuous

        if self.regression_type == "GE":
            if self.noise == "Gaussian":
                return F.mse_loss(preds, yb)
            elif self.noise == "Cauchy":
                # −log p(y|φ,γ) where p is Cauchy with scale γ. We learn a
                # global logγ parameter.
                if not hasattr(self, "log_gamma"):
                    self.log_gamma = nn.Parameter(torch.zeros(1, device=self.device))
                gamma = torch.exp(self.log_gamma)
                return (
                    torch.log(math.pi * gamma)
                    + torch.log(1 + ((yb - preds) / gamma) ** 2).mean()
                )
            else:
                raise NotImplementedError("Only Gaussian or Cauchy noise implemented.")
        else:  # MPA – classification over *num_tasks* categories
            return F.cross_entropy(preds, yb.long())

    def _predict_continuous(self, xb: torch.Tensor) -> torch.Tensor:
        """Forward pass returning *continuous* latent phenotypes, with optional GE
        non‑linearity, followed by task‑specific linear head. Used by both GE
        and MPA (where logits are derived from continuous φ)."""
        xb = xb.to(self.device)
        z = self.gpmap(xb)  # (N,1)
        z_nl = self.nonlinearity(z)  # (N,1)
        out = self.head(z_nl)  # (N,num_tasks)
        return out

    @torch.no_grad()
    def get_info(self, verbose: int = 1) -> float:
        """Compute a *heuristic* predictive information metric.  For GE we return
        the test‑set R².  For MPA we return accuracy.  This does *not* attempt
        to reproduce the variational information bound used by MAVE‑NN, but is
        often a useful quick‑and‑dirty proxy."""
        self.eval()
        all_preds, all_y = [], []
        for xb, yb in self.test_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            preds = self._predict_continuous(xb)
            all_preds.append(preds.cpu())
            all_y.append(yb.cpu())
        y_true = torch.cat(all_y, dim=0)
        y_pred = torch.cat(all_preds, dim=0)

        if self.regression_type == "GE":
            ss_res = ((y_true - y_pred) ** 2).sum()
            ss_tot = ((y_true - y_true.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot
            if verbose:
                print(f"Test R²: {r2:.4f}")
            return r2.item()
        else:  # classification accuracy
            acc = (y_pred.argmax(dim=1) == y_true.long()).float().mean()
            if verbose:
                print(f"Test accuracy: {acc:.4f}")
            return acc.item()

    def get_params(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return (theta_0, theta_lc, theta_lclc). Pairwise parameters are only
        returned if gpmap_type == 'pairwise'."""
        theta_0 = self.gpmap.theta_0.detach().cpu().numpy()
        theta_lc = self.gpmap.theta_lc.detach().cpu().numpy()
        theta_lclc = None
        if self.gpmap_type == "pairwise":
            # Reconstruct full L×A×L×A tensor from low‑rank factors
            U = self.gpmap.U.detach().cpu().numpy()  # (L,A,K)
            K = U.shape[-1]
            # θ_{l1,c1,l2,c2} ≈ Σ_k U_{l1,c1,k} U_{l2,c2,k}
            theta_lclc = np.einsum("lak, mbk -> lamb", U, U)  # shape (L,A,L,A)
        return theta_0, theta_lc, theta_lclc
