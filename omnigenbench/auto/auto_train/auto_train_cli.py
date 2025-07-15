# -*- coding: utf-8 -*-
# file: auto_bench_cli.py
# time: 19:18 05/02/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import argparse
import os
import platform
import sys
import time

from typing import Optional

# Handle both relative and absolute imports
try:
    from ..auto_train.auto_train import AutoTrain
    from ...src.misc.utils import fprint
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from omnigenbench.auto.auto_train.auto_train import AutoTrain
    from omnigenbench.src.misc.utils import fprint


def train_command(args: Optional[list] = None):
    """
    Entry point for the OmniGenome auto-train command-line interface.

    :param args: A list of command-line arguments. If None, `sys.argv` is used.
    """

    parser = create_parser()
    parsed_args = parser.parse_args(args)

    model_path = parsed_args.model
    fprint(f"\n>> Starting evaluation for model: {model_path}")

    # Special handling for multimolecule models
    if "multimolecule" in model_path:
        from multimolecule import RnaTokenizer, AutoModelForTokenPrediction

        tokenizer = RnaTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenPrediction.from_pretrained(
            model_path, trust_remote_code=True
        ).base_model
    else:
        tokenizer = parsed_args.tokenizer
        model = model_path

    # Initialize AutoTraining
    autobench = AutoTrain(
        dataset=parsed_args.dataset,
        model_name_or_path=model,
        tokenizer=tokenizer,
        overwrite=parsed_args.overwrite,
        trainer=parsed_args.trainer,
    )

    # Run evaluation
    autobench.run(**vars(parsed_args))


def create_parser() -> argparse.ArgumentParser:
    """
    Creates the argument parser for the auto-train CLI.

    :return: An `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Genomic Foundation Model Benchmark Suite (Single Model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required argument
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Path to the dataset and training configuration file.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default=None,
        help="Path to the tokenizer to use (HF tokenizer ID or local path).",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the model to evaluate (HF model ID or local path).",
    )

    # Optional arguments
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite existing bench results, otherwise resume from training checkpoint.",
    )
    parser.add_argument(
        "--bs_scale",
        type=int,
        default=1,
        help="Batch size scale factor. To increase GPU memory utilization, set to 2 or 4, etc.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="accelerate",
        choices=["native", "accelerate", "hf_trainer"],
        help="Trainer to use for training. \n"
        "Use 'accelerate' for distributed training. Set to false to disable. "
        "You can use 'accelerate config' to customize behavior.\n"
        "Use 'hf_trainer' for Hugging Face Trainer. \n"
        "Set to 'native' to use native PyTorch training loop.\n",
    )
    parser.add_argument(
        "--autocast",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16", "fp8", "no"],
        help="Automatic mixed precision training mode.",
    )
    return parser


def run_train():
    """
    Wrapper function to run the auto-train command.

    This function is the entry point for the 'autotrain' console script.
    """
    try:
        train_command()
    except Exception as e:
        fprint(f"Error running auto-train: {e}")
        sys.exit(1)


if __name__ == "__main__":
    train_command()
