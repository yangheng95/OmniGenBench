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
from typing import Optional

# Handle both relative and absolute imports
try:
    from ..auto_train.auto_train import AutoTrain
    from ...src.misc.utils import fprint, load_module_from_path
    from ..config.auto_config import AutoConfig
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
    parsed_args, unknown_args = parser.parse_known_args(args)
    # Dynamically load config.py if it exists in the dataset directory
    config_path = os.path.join(parsed_args.dataset, "config.py")
    print(f"Loading configuration from: {config_path}")
    if os.path.exists(config_path):
        config = load_module_from_path("module_name", config_path)
        for attr_name in dir(config):
            attr = getattr(config, attr_name)
            if isinstance(attr, AutoConfig):  # Check if it is an instance of AutoConfig
                # Process the found AutoConfig instance
                for key, value in vars(
                    attr
                ).items():  # Iterate over all attributes of the instance
                    if not hasattr(parsed_args, key):
                        setattr(parsed_args, key, value)

    # Convert unknown arguments into a dictionary
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            value = True  # Default to True for flags
            if "=" in key:
                key, value = key.split("=", 1)
            extra_args[key] = value

    # Merge extra_args into parsed_args
    for key, value in extra_args.items():
        if not hasattr(parsed_args, key):
            setattr(parsed_args, key, value)

    model_path = parsed_args.model
    fprint(f"\n>> Starting training for model: {model_path}")

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
    args = vars(parsed_args)
    args.pop("model")
    args.pop("tokenizer")
    autotrain = AutoTrain(
        dataset=args.pop("dataset"),
        model_name_or_path=model,
        tokenizer=tokenizer,
        overwrite=args.pop("overwrite", False),
        trainer=args.pop("trainer", "accelerate"),
        **vars(parsed_args),  # Pass all parsed arguments
    )

    # Run training
    autotrain.run(**vars(parsed_args))


def create_parser() -> argparse.ArgumentParser:
    """
    Creates the argument parser for the auto-train CLI.

    :return: An `argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description="Genomic Foundation Model Training Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required arguments
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
        help="Overwrite existing training results, otherwise resume from checkpoint.",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="native",
        choices=["native", "accelerate", "hf_trainer"],
        help="Trainer to use for training. Options: native, accelerate, hf_trainer.",
    )
    return parser


def run_train():
    """
    Wrapper function to run the auto-train command.

    This function is the entry point for the 'autotrain' console script.
    """
    train_command()


if __name__ == "__main__":
    train_command()
