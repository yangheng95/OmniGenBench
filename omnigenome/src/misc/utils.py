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


class RNA2StructureCache(dict):
    """
    A cache for RNA sequence to structure predictions using ViennaRNA.
    
    This class provides a dictionary-like interface for caching RNA secondary
    structure predictions. It uses ViennaRNA for structure prediction and
    supports both single sequences and batches of sequences.
    
    The cache can be persisted to disk and loaded back, making it useful for
    avoiding redundant structure predictions across multiple runs.
    
    Attributes:
        cache_file (str): Path to the cache file on disk.
        cache (dict): The in-memory cache dictionary.
        queue_num (int): Counter for tracking cache updates.
    """

    def __init__(self, cache_file=None, *args, **kwargs):
        """
        Initializes the RNA structure cache.

        Args:
            cache_file (str, optional): Path to the cache file. If None, uses
                                      a default path in `__OMNIGENOME_DATA__`.
            *args: Additional arguments passed to dict constructor.
            **kwargs: Additional keyword arguments passed to dict constructor.

        Example:
            >>> # Initialize with default cache file
            >>> cache = RNA2StructureCache()
            
            >>> # Initialize with custom cache file
            >>> cache = RNA2StructureCache("my_cache.pkl")
        """
        super().__init__(*args, **kwargs)

        if not cache_file:
            self.cache_file = "__OMNIGENOME_DATA__/rna2structure.cache.pkl"
        else:
            self.cache_file = cache_file

        if self.cache_file is None or not os.path.exists(self.cache_file):
            self.cache = {}
        else:
            fprint(f"Initialize sequence to structure cache from {self.cache_file}...")
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)

        self.queue_num = 0

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

        if (
            os.name != "nt" and len(sequences) > 1
        ):  # multiprocessing is not working on Windows in my case
            num_workers = min(os.cpu_count(), len(sequences))

        structures = []

        if not all([seq in self.cache for seq in sequences]):
            if num_workers == 1:
                for seq in sequences:
                    if seq not in self.cache:
                        self.queue_num += 1
                        self.cache[seq] = RNA.fold(seq)
            else:
                if num_workers is None:
                    num_workers = min(os.cpu_count(), len(sequences))

                with multiprocessing.Pool(num_workers) as pool:
                    for seq in sequences:
                        if seq not in self.cache:
                            self.queue_num += 1
                            async_result = pool.apply_async(RNA.fold, args=(seq,))
                            structures.append((seq, async_result))

                    for seq, result in structures:
                        self.cache[seq] = result.get()  # result is a tuple

        if return_mfe:
            structures = [self.cache[seq] for seq in sequences]
        else:
            structures = [self.cache[seq][0] for seq in sequences]
        self.update_cache_file(self.cache_file)

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

        if not os.path.exists(os.path.dirname(cache_file)):
            os.makedirs(os.path.dirname(cache_file))

        # print(f"Updating cache file {cache_file}...")
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache, f)

        self.queue_num = 0


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
    args = [key for key in sorted(config.args.keys())]
    if logger:
        logger.info(args)
    else:
        fprint(args)


def fprint(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
    """
    Custom print function that adds a timestamp and the pyabsa version before the printed message.

    Args:
        *objects: Any number of objects to be printed
        sep (str, optional): Separator between objects. Defaults to " ".
        end (str, optional): Ending character after all objects are printed. Defaults to "\n".
        file (io.TextIOWrapper, optional): Text file to write printed output to. Defaults to sys.stdout.
        flush (bool, optional): Whether to flush output buffer after printing. Defaults to False.
    """
    from omnigenome import __version__
    from omnigenome import __name__

    print(
        time.strftime(
            "[%Y-%m-%d %H:%M:%S] [{} {}] ".format(__name__, __version__),
            time.localtime(time.time()),
        ),
        *objects,
        sep=sep,
        end=end,
        file=file,
        flush=flush,
    )


def clean_temp_checkpoint(days_threshold=7):
    """
    删除超过指定时间的 checkpoint 文件。

    参数：
    - directory (str): 文件所在的目录路径。
    - file_extension (str): checkpoint 文件的扩展名，默认是 ".ckpt"。
    - days_threshold (int): 超过多少天的文件将被删除，默认是 7 天。
    """
    # 获取当前时间
    import os
    from datetime import datetime, timedelta

    current_time = datetime.now()
    ckpt_files = findfile.find_cwd_files(["tmp_ckpt", ".pt"])
    # 遍历目录中的所有文件
    for file_path in ckpt_files:
        # 获取文件的最后修改时间
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

        # 计算文件是否超过指定的时间阈值
        if current_time - file_mod_time > timedelta(days=days_threshold):
            try:
                # 删除文件
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def load_module_from_path(module_name, file_path):
    import importlib

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ImportError(f"Cannot find the module {module_name} from {file_path}.")
    return module


def check_bench_version(bench_version, omnigenome_version):
    assert (
        bench_version is not None
    ), "Benchmark metadata does not contain a valid __omnigenome__ version."

    if not isinstance(bench_version, (int, float, str)):
        raise TypeError(
            f"Invalid type for benchmark version. Expected int, float, or str but got {type(bench_version).__name__}."
        )

    assert (
        omnigenome_version is not None
    ), "AutoBench is missing a valid omnigenome version."

    if bench_version > omnigenome_version:
        raise ValueError(
            f"AutoBench version {omnigenome_version} is not compatible with the benchmark version "
            f"{bench_version}. Please update the benchmark or AutoBench."
        )


def clean_temp_dir_pt_files():
    tmp_dir = tempfile.gettempdir()
    for f in os.listdir(tmp_dir):
        if f.endswith(".pt") and f.startswith("tmp_ckpt"):
            path = os.path.join(tmp_dir, f)
            try:
                os.remove(path)
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
