# -*- coding: utf-8 -*-
# file: utils.py
# time: 14:45 06/04/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2024. All Rights Reserved.
import multiprocessing
import os
import pickle
import sys
import tempfile
import time
import warnings

import ViennaRNA as RNA
import findfile

default_omnigenome_repo = (
    "https://huggingface.co/spaces/yangheng/OmniGenomeLeaderboard/"
)


def seed_everything(seed=42):
    """
    Sets random seeds for reproducibility across all random number generators.

    This function sets seeds for Python's random module, NumPy, PyTorch (CPU and CUDA),
    and sets the PYTHONHASHSEED environment variable to ensure reproducible results
    across different runs.

    Args:
        seed (int): The seed value to use for all random number generators.
                   Defaults to 42.

    Example:
        >>> # Set seeds for reproducibility
        >>> seed_everything(42)
        >>> # Now all random operations will be reproducible
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RNA2StructureCache(dict):
    """
    A cache for RNA secondary structure predictions using ViennaRNA.

    This class provides a caching mechanism for RNA secondary structure predictions
    to avoid redundant computations. It supports both single sequence and batch
    processing with optional multiprocessing for improved performance.

    Attributes:
        cache (dict): Dictionary storing sequence-structure mappings
        cache_file (str): Path to the cache file on disk
        queue_num (int): Counter for tracking cache updates
    """

    def __init__(self, cache_file=None, *args, **kwargs):
        """
        Initialize the RNA structure cache.

        Args:
            cache_file (str, optional): Path to the cache file. If None, uses
                                      a default temporary file.
            *args: Additional positional arguments for dict initialization
            **kwargs: Additional keyword arguments for dict initialization
        """
        super().__init__(*args, **kwargs)
        self.cache = dict(*args, **kwargs)
        self.cache_file = (
            cache_file
            if cache_file is not None
            else os.path.join(tempfile.gettempdir(), "rna_structure_cache.pkl")
        )
        self.queue_num = 0

        # Load existing cache if available
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self.cache.update(pickle.load(f))
            except Exception as e:
                warnings.warn(f"Failed to load cache file: {e}")

    def __getitem__(self, key):
        """Gets a cached structure prediction."""
        return self.cache[key]

    def __setitem__(self, key, value):
        """Sets a structure prediction in the cache."""
        self.cache[key] = value

    def __str__(self):
        """String representation of the cache."""
        return str(self.cache)

    def __repr__(self):
        """String representation of the cache."""
        return str(self.cache)

    def _fold_single_sequence(self, sequence):
        """
        Predict structure for a single sequence (worker function for multiprocessing).

        Args:
            sequence (str): RNA sequence to fold

        Returns:
            tuple: (structure, mfe) tuple
        """
        try:
            return RNA.fold(sequence)
        except Exception as e:
            warnings.warn(f"Failed to fold sequence {sequence}: {e}")
            return ("." * len(sequence), 0.0)

    def fold(self, sequence, return_mfe=False, num_workers=1):
        """
        Predicts RNA secondary structure for given sequences.

        This method predicts RNA secondary structures using ViennaRNA. It supports
        both single sequences and batches of sequences. The method uses caching
        to avoid redundant predictions and supports multiprocessing for batch
        processing on non-Windows systems.

        Args:
            sequence (str or list): A single RNA sequence or a list of sequences.
            return_mfe (bool): Whether to return minimum free energy along with
                              structure. Defaults to False.
            num_workers (int): Number of worker processes for batch processing.
                             Defaults to 1. Set to None for auto-detection.

        Returns:
            str or list: The predicted structure(s). If return_mfe is True,
                        returns tuples of (structure, mfe).

        Example:
            >>> cache = RNA2StructureCache()
            >>> # Predict structure for a single sequence
            >>> structure = cache.fold("GGGAAAUCC")
            >>> print(structure)  # "(((...)))"

            >>> # Predict structures for multiple sequences
            >>> structures = cache.fold(["GGGAAAUCC", "AUUGCUAA"])
            >>> print(structures)  # ["(((...)))", "........"]
        """
        if not isinstance(sequence, list):
            sequences = [sequence]
        else:
            sequences = sequence

        # Determine if we should use multiprocessing
        use_multiprocessing = (
            os.name != "nt"  # Not Windows
            and len(sequences) > 1  # Multiple sequences
            and num_workers > 1  # Multiple workers requested
        )

        # Find sequences that need prediction
        sequences_to_predict = [seq for seq in sequences if seq not in self.cache]

        if sequences_to_predict:
            if use_multiprocessing:
                # Use multiprocessing for batch prediction
                if num_workers is None:
                    num_workers = min(os.cpu_count(), len(sequences_to_predict))

                try:
                    # Set multiprocessing start method to 'spawn' for better compatibility
                    if multiprocessing.get_start_method(allow_none=True) != "spawn":
                        multiprocessing.set_start_method("spawn", force=True)

                    with multiprocessing.Pool(num_workers) as pool:
                        # Use map instead of apply_async for better error handling
                        results = pool.map(
                            self._fold_single_sequence, sequences_to_predict
                        )

                        # Update cache with results
                        for seq, result in zip(sequences_to_predict, results):
                            self.cache[seq] = result
                            self.queue_num += 1

                except Exception as e:
                    warnings.warn(
                        f"Multiprocessing failed, falling back to sequential: {e}"
                    )
                    # Fallback to sequential processing
                    for seq in sequences_to_predict:
                        self.cache[seq] = self._fold_single_sequence(seq)
                        self.queue_num += 1
            else:
                # Sequential processing
                for seq in sequences_to_predict:
                    self.cache[seq] = self._fold_single_sequence(seq)
                    self.queue_num += 1

        # Prepare output
        if return_mfe:
            structures = [self.cache[seq] for seq in sequences]
        else:
            structures = [self.cache[seq][0] for seq in sequences]

        # Update cache file periodically
        self.update_cache_file(self.cache_file)

        # Return single result or list
        if len(structures) == 1:
            return structures[0]
        else:
            return structures

    def update_cache_file(self, cache_file=None):
        """
        Updates the cache file on disk.

        This method saves the in-memory cache to disk. It only saves when
        the queue_num reaches 100 to avoid excessive disk I/O.

        Args:
            cache_file (str, optional): Path to the cache file. If None, uses
                                      the instance's cache_file.

        Example:
            >>> cache.update_cache_file()  # Force save to disk
        """
        if self.queue_num < 100:
            return

        if cache_file is None:
            cache_file = self.cache_file

        try:
            if not os.path.exists(os.path.dirname(cache_file)):
                os.makedirs(os.path.dirname(cache_file))

            with open(cache_file, "wb") as f:
                pickle.dump(self.cache, f)

            self.queue_num = 0
        except Exception as e:
            warnings.warn(f"Failed to update cache file: {e}")


def env_meta_info():
    """
    Collects metadata about the current environment and library versions.

    This function gathers information about the current Python environment,
    including versions of key libraries like PyTorch and Transformers,
    as well as OmniGenome version information.

    Returns:
        dict: A dictionary containing environment metadata including:
              - library_name: Name of the OmniGenome library
              - omnigenome_version: Version of OmniGenome
              - torch_version: PyTorch version with CUDA info
              - transformers_version: Transformers library version

    Example:
        >>> metadata = env_meta_info()
        >>> print(metadata['torch_version'])  # "2.0.0+cu118+git..."
    """
    from torch.version import __version__ as torch_version
    from torch.version import cuda as torch_cuda_version
    from torch.version import git_version
    from transformers import __version__ as transformers_version
    from ... import __version__ as omnigenome_version
    from ... import __name__ as omnigenome_name

    return {
        "library_name": omnigenome_name,
        "omnigenome_version": omnigenome_version,
        "torch_version": f"{torch_version}+cu{torch_cuda_version}+git{git_version}",
        "transformers_version": transformers_version,
    }


def naive_secondary_structure_repair(sequence, structure):
    """
    Repair the secondary structure of a sequence.

    This function attempts to repair malformed RNA secondary structure
    representations by ensuring proper bracket matching. It handles
    common issues like unmatched brackets by converting them to dots.

    Args:
        sequence (str): A string representing the sequence.
        structure (str): A string representing the secondary structure.

    Returns:
        str: A string representing the repaired secondary structure.

    Example:
        >>> sequence = "GGGAAAUCC"
        >>> structure = "(((...)"  # Malformed structure
        >>> repaired = naive_secondary_structure_repair(sequence, structure)
        >>> print(repaired)  # "(((...))"
    """
    repaired_structure = ""
    stack = []
    for i, (s, c) in enumerate(zip(structure, sequence)):
        if s == "(":
            stack.append(i)
        elif s == ")":
            if stack:
                stack.pop()
            else:
                repaired_structure += "."
        else:
            repaired_structure += s
    for i in stack:
        repaired_structure = repaired_structure[:i] + "." + repaired_structure[i + 1 :]
    return repaired_structure


def save_args(config, save_path):
    """
    Save arguments to a file.

    This function saves the arguments from a configuration object to a text file.
    It's useful for logging experiment parameters and configurations.

    Args:
        config: A Namespace object containing the arguments.
        save_path (str): A string representing the path of the file to be saved.

    Example:
        >>> from argparse import Namespace
        >>> config = Namespace(learning_rate=0.001, batch_size=32)
        >>> save_args(config, "config.txt")
    """
    f = open(os.path.join(save_path), mode="w", encoding="utf8")
    for arg in config.args:
        if config.args_call_count[arg]:
            f.write("{}: {}\n".format(arg, config.args[arg]))
    f.close()


def print_args(config, logger=None):
    """
    Print the arguments to the console.

    This function prints the arguments from a configuration object to the console
    or a logger. It's useful for debugging and logging experiment parameters.

    Args:
        config: A Namespace object containing the arguments.
        logger: A logger object. If None, prints to console.

    Example:
        >>> from argparse import Namespace
        >>> config = Namespace(learning_rate=0.001, batch_size=32)
        >>> print_args(config)
    """
    if logger is None:
        for arg in config.args:
            if config.args_call_count[arg]:
                print("{}: {}".format(arg, config.args[arg]))
    else:
        for arg in config.args:
            if config.args_call_count[arg]:
                logger.info("{}: {}".format(arg, config.args[arg]))


def fprint(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
    """
    Enhanced print function with automatic flushing.

    This function provides a print-like interface with automatic flushing
    to ensure output is displayed immediately. It's useful for real-time
    logging and progress tracking.

    Args:
        *objects: Objects to print
        sep (str): Separator between objects (default: " ")
        end (str): String appended after the last value (default: "\n")
        file: File-like object to write to (default: sys.stdout)
        flush (bool): Whether to flush the stream (default: False)

    Example:
        >>> fprint("Training started...", flush=True)
        >>> fprint("Epoch 1/10", "Loss: 0.5", sep=" | ")
    """
    print(*objects, sep=sep, end=end, file=file, flush=True)


def clean_temp_checkpoint(days_threshold=7):
    """
    Clean up temporary checkpoint files older than specified days.

    This function removes temporary checkpoint files that are older than
    the specified threshold to free up disk space.

    Args:
        days_threshold (int): Number of days after which files are considered old.
                            Defaults to 7.

    Example:
        >>> clean_temp_checkpoint(3)  # Remove files older than 3 days
    """
    import glob
    import time

    temp_patterns = [
        "temp_checkpoint_*",
        "checkpoint_*",
        "*.tmp",
        "*.temp",
    ]

    current_time = time.time()
    threshold_time = current_time - (days_threshold * 24 * 60 * 60)

    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.getmtime(file_path) < threshold_time:
                    os.remove(file_path)
            except Exception:
                pass


def load_module_from_path(module_name, file_path):
    """
    Load a Python module from a file path.

    This function dynamically loads a Python module from a file path,
    useful for loading configuration files or custom modules.

    Args:
        module_name (str): Name to assign to the loaded module
        file_path (str): Path to the Python file to load

    Returns:
        module: The loaded module object

    Example:
        >>> config = load_module_from_path("config", "config.py")
        >>> print(config.some_variable)
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_bench_version(bench_version, omnigenome_version):
    """
    Check if benchmark version is compatible with OmniGenome version.

    This function compares the benchmark version with the OmniGenome version
    to ensure compatibility and warns if there are potential issues.

    Args:
        bench_version (str): Version of the benchmark
        omnigenome_version (str): Version of OmniGenome

    Example:
        >>> check_bench_version("0.2.0", "0.3.0")
    """
    if bench_version != omnigenome_version:
        warnings.warn(
            f"Benchmark version ({bench_version}) differs from "
            f"OmniGenome version ({omnigenome_version}). "
            f"This may cause compatibility issues."
        )


def clean_temp_dir_pt_files():
    """
    Clean up temporary PyTorch files in the current directory.

    This function removes temporary PyTorch files (like .pt, .pth files)
    that may be left over from previous runs.

    Example:
        >>> clean_temp_dir_pt_files()
    """
    import glob

    temp_patterns = ["*.pt", "*.pth", "temp_*", "checkpoint_*"]

    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception:
                pass
