# -*- coding: utf-8 -*-
# file: model.py
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
"""
RNA design model using masked language modeling and evolutionary algorithms.

This module provides an RNA design model that combines masked language modeling
with evolutionary algorithms to design RNA sequences that fold into specific
target structures. It uses a multi-objective optimization approach to balance
structure similarity and thermodynamic stability.
"""
import random
import numpy as np
import torch
import autocuda
from transformers import AutoModelForMaskedLM, AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
import ViennaRNA
from scipy.spatial.distance import hamming
import warnings
import os

from ....src.misc.utils import fprint


class OmniModelForRNADesign(torch.nn.Module):
    """
    RNA design model using masked language modeling and evolutionary algorithms.

    This model combines a pre-trained masked language model with evolutionary
    algorithms to design RNA sequences that fold into specific target structures.
    It uses a multi-objective optimization approach to balance structure similarity
    and thermodynamic stability.

    Attributes:
        device: Device to run the model on (CPU or GPU)
        parallel: Whether to use parallel processing for structure prediction
        tokenizer: Tokenizer for processing RNA sequences
        model: Pre-trained masked language model
    """

    def __init__(
        self,
        model="yangheng/OmniGenome-186M",
        device=None,
        parallel=False,
        *args,
        **kwargs,
    ):
        """
        Initialize the RNA design model.

        Args:
            model (str): Model name or path for the pre-trained MLM model
            device: Device to run the model on (default: None, auto-detect)
            parallel (bool): Whether to use parallel processing (default: False)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.device = autocuda.auto_cuda() if device is None else device
        self.parallel = parallel
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model, trust_remote_code=True)
        # prefer float16 on CUDA
        try:
            self.model.to(self.device)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                self.model.to(torch.float16)
        except Exception:
            self.model.to(self.device)

    # --------------------------
    # Folding helpers
    # --------------------------
    @staticmethod
    def _random_bp_span(bp_span):
        """
        Sample a bp span around the provided bp_span, bounded to <=400.
        """
        try:
            return random.choice(
                list(range(max(0, int(bp_span) - 50), min(int(bp_span) + 50, 400)))
            )
        except Exception:
            return 400

    @staticmethod
    def _longest_bp_span(structure):
        """
        Compute the longest base-pair distance using a stack over dot-bracket notation.
        """
        stack = []
        max_span = 0
        for i, ch in enumerate(structure):
            if ch == "(":  # push index of left bracket
                stack.append(i)
            elif ch == ")" and stack:
                left = stack.pop()
                max_span = max(max_span, i - left)
        return max_span if max_span > 0 else len(structure)

    @staticmethod
    def _predict_structure_single(sequence, bp_span=-1):
        """
        Predict structure and MFE for a single sequence using ViennaRNA with bp-span control.
        """
        try:
            md = ViennaRNA.md()
            # ensure reasonably large search space
            md.max_bp_span = max(
                OmniModelForRNADesign._random_bp_span(
                    bp_span if bp_span != -1 else len(sequence)
                ),
                400,
            )
            fc = ViennaRNA.fold_compound(sequence, md)
            ss, mfe = fc.mfe()
            return ss, mfe
        except Exception as e:
            warnings.warn(f"Failed to fold sequence {sequence}: {e}")
            return ("." * len(sequence), 0.0)

    # --------------------------
    # Evolutionary operators
    # --------------------------
    def _init_population(self, structure, num_population):
        """
        Initialize the population via MLM conditioned on target structure.
        """
        population = []
        mlm_inputs = []
        L = len(structure)
        for _ in range(num_population):
            # biased GC seeding with masks
            masked_sequence = [
                random.choice(["G", "C", self.tokenizer.mask_token]) for _ in range(L)
            ]
            masked_sequence_str = "".join(masked_sequence)
            mlm_inputs.append(f"{masked_sequence_str}<eos>{''.join(structure)}")
        outputs = self._mlm_predict(mlm_inputs, structure)
        for i in range(outputs.size(0)):
            toks = self.tokenizer.convert_ids_to_tokens(outputs[i].tolist())
            # reconstruct: only fill masked positions with predicted base if valid
            sentinel_input = mlm_inputs[i].replace(self.tokenizer.mask_token, "$")
            fixed = [
                (
                    x
                    if (x and x in "AGCT" and y == "$")
                    else (
                        y if (y and y != "$") else random.choice(["A", "T", "G", "C"])
                    )
                )
                for x, y in zip(toks, list(sentinel_input[:L]))
            ]
            bp_span = self._random_bp_span(L)
            population.append(("".join(fixed), bp_span))
        return population

    def _mlm_mutate(self, population, structure, mutation_ratio):
        """
        Mutate population using MLM prompts and per-position masking.
        """

        def mutate_string(seq, rate):
            arr = np.array(list(seq), dtype=np.str_)
            mask = np.random.rand(*arr.shape) < rate
            arr[mask] = "$"
            return "".join(arr.tolist()).replace("$", self.tokenizer.mask_token)

        mlm_inputs = []
        masked_sequences = []
        pop_size = len(population)
        # select parents (here from whole population)
        for i in range(pop_size):
            seq, _bp = population[random.randrange(pop_size)]
            masked = mutate_string(seq, mutation_ratio)
            masked_sequences.append(masked)
            mlm_inputs.append(f"{masked}<eos>{''.join(structure)}")
        outputs = self._mlm_predict(mlm_inputs, structure)
        mut_population = []
        for i in range(outputs.size(0)):
            toks = self.tokenizer.convert_ids_to_tokens(outputs[i].tolist())
            sentinel = masked_sequences[i].replace(self.tokenizer.mask_token, "$")
            fixed = [
                (
                    x
                    if (x and x in "AGCT" and y == "$")
                    else (
                        y if (y and y != "$") else random.choice(["A", "T", "G", "C"])
                    )
                )
                for x, y in zip(toks, list(sentinel[: len(structure)]))
            ]
            # inherit and jitter parent bp span
            _, parent_bp = population[i % pop_size]
            mut_population.append(("".join(fixed), self._random_bp_span(parent_bp)))
        return mut_population

    def _crossover(self, population, num_points=3):
        """
        Multi-point crossover between randomly chosen parents.
        """
        if len(population) < 2:
            return population
        seqs = [seq for seq, _ in population]
        pop_size = len(seqs)
        L = len(seqs[0])
        parents_idx = np.random.choice(pop_size, (pop_size, 2))
        p1 = np.array([list(seqs[i]) for i in parents_idx[:, 0]])
        p2 = np.array([list(seqs[i]) for i in parents_idx[:, 1]])
        cps = np.sort(np.random.randint(1, L, size=(pop_size, num_points)), axis=1)
        masks = np.zeros((pop_size, L), dtype=bool)
        for i in range(pop_size):
            last = 0
            for j in range(num_points):
                masks[i, last : cps[i, j]] = j % 2 == 0
                last = cps[i, j]
            masks[i, last:] = num_points % 2 == 0
        child1 = np.where(masks, p1, p2)
        child2 = np.where(masks, p2, p1)
        # keep bp_span from corresponding slot
        children = [("".join(child1[i]), population[i][1]) for i in range(pop_size)] + [
            ("".join(child2[i]), population[i][1]) for i in range(pop_size)
        ]
        return children

    # --------------------------
    # Evaluation and selection
    # --------------------------
    @staticmethod
    def _non_dominated_sorting(scores, mfe_values):
        num_solutions = len(scores)
        domination_count = [0] * num_solutions
        dominated_solutions = [[] for _ in range(num_solutions)]
        fronts = [[]]
        for p in range(num_solutions):
            for q in range(num_solutions):
                if scores[p] < scores[q] and mfe_values[p] < mfe_values[q]:
                    dominated_solutions[p].append(q)
                elif scores[q] < scores[p] and mfe_values[q] < mfe_values[p]:
                    domination_count[p] += 1
            if domination_count[p] == 0:
                fronts[0].append(p)
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        if not fronts[-1]:
            fronts.pop(-1)
        return fronts

    @staticmethod
    def _select_next_generation(next_generation, fronts):
        sorted_population = []
        for front in fronts:
            front_population = [next_generation[i] for i in front]
            sorted_population.extend(front_population)
            if len(sorted_population) >= len(next_generation):
                break
        return sorted_population[: len(next_generation)]

    def _evaluate_structure_fitness(self, sequences, structure):
        """
        Evaluate sequences by folding and computing hamming distance to target structure; select via NSGA-like fronts.
        """
        # parallel folding with order preserved
        results = [None] * len(sequences)
        try:
            max_workers = min(os.cpu_count() or 1, len(sequences), 8)
            if self.parallel and len(sequences) > 1 and max_workers > 1:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = {
                        ex.submit(
                            OmniModelForRNADesign._predict_structure_single,
                            seq,
                            bp_span,
                        ): idx
                        for idx, (seq, bp_span) in enumerate(sequences)
                    }
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            results[idx] = fut.result()
                        except Exception as e:
                            warnings.warn(f"Failed to process sequence at {idx}: {e}")
                            seq, _ = sequences[idx]
                            results[idx] = ("." * len(seq), 0.0)
            else:
                for idx, (seq, bp_span) in enumerate(sequences):
                    results[idx] = self._predict_structure_single(seq, bp_span)
        except Exception as e:
            warnings.warn(
                f"Parallel processing failed, falling back to sequential: {e}"
            )
            for idx, (seq, bp_span) in enumerate(sequences):
                try:
                    results[idx] = self._predict_structure_single(seq, bp_span)
                except Exception as ie:
                    warnings.warn(f"Failed to fold sequence {seq}: {ie}")
                    results[idx] = ("." * len(seq), 0.0)
        # build scored population
        scored = []
        for (seq, bp_span), (ss, mfe) in zip(sequences, results):
            score = hamming(list(structure), list(ss))
            scored.append((seq, bp_span, score, mfe))
        fronts = self._non_dominated_sorting(
            [x[2] for x in scored], [x[3] for x in scored]
        )
        # additionally sort within fronts by score for stability
        selected = self._select_next_generation(scored, fronts)
        selected = sorted(selected, key=lambda x: x[2])
        return selected

    # --------------------------
    # MLM inference
    # --------------------------
    def _mlm_predict(self, mlm_inputs, structure):
        """
        Tokenize batch of prompts and get argmax token ids. Returns shape [B, len(structure)]
        """
        batch_size = 16
        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(mlm_inputs), batch_size):
                batch = self.tokenizer(
                    mlm_inputs[i : i + batch_size],
                    padding=False,
                    max_length=1024,
                    truncation=True,
                    return_tensors="pt",
                )
                batch = {
                    k: v.to(torch.int64).to(self.model.device) for k, v in batch.items()
                }
                if torch.cuda.is_available() and str(self.model.device).startswith(
                    "cuda"
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        logits = self.model(**batch)[0]
                else:
                    logits = self.model(**batch)[0]
                preds = logits.argmax(dim=-1)
                all_outputs.append(preds)
                # free
                del batch
                del logits
        outputs = torch.cat(all_outputs, dim=0)
        return outputs[:, 1 : 1 + len(structure)]

    # --------------------------
    # Public API
    # --------------------------
    def design(
        self, structure, mutation_ratio=0.5, num_population=100, num_generation=100
    ):
        """
        Design RNA sequences for a target structure using evolutionary algorithms.
        """
        # init
        population = self._init_population(structure, num_population)
        population = self._mlm_mutate(population, structure, mutation_ratio)
        # evolve
        for _ in range(num_generation):
            next_generation = self._crossover(population)
            next_generation = self._mlm_mutate(
                next_generation, structure, mutation_ratio
            )
            next_generation = self._evaluate_structure_fitness(
                next_generation, structure
            )[:num_population]
            # early stop
            candidates = [
                seq for seq, _bp, score, _mfe in next_generation if score == 0
            ]
            if candidates:
                return candidates
            population = [
                (seq, bp_span) for seq, bp_span, _score, _mfe in next_generation
            ]
        # fallback: return the best sequence encountered in last population
        return population[0][0]


# Example usage
if __name__ == "__main__":
    model = OmniModelForRNADesign(model="anonymous8/OmniGenome-186M")
    best_sequence = model.design(
        structure="(((....)))",
        mutation_ratio=0.5,
        num_population=100,
        num_generation=100,
    )
    fprint(f"Best RNA sequence: {best_sequence}")
