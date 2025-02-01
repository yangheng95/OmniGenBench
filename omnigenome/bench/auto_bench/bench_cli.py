# -*- coding: utf-8 -*-
# file: bench_cli.py
# time: 21:06 31/01/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (Yang Heng)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import argparse
import os
import platform
import subprocess
import sys
import time
from typing import Optional

from omnigenome import AutoBench
from omnigenome.src.misc.utils import fprint



def bench_command(args: Optional[list] = None):
    """Entry function for BEACON benchmark testing (single model version)."""

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
        tokenizer = None
        model = model_path

    # Initialize benchmark
    autobench = AutoBench(
        benchmark=parsed_args.benchmark,
        model_name_or_path=model,
        tokenizer=tokenizer,
        overwrite=parsed_args.overwrite,
        use_hf_trainer=parsed_args.use_hf_trainer,
    )

    # Run evaluation
    autobench.run(
        **vars(parsed_args)
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser (single model version)."""
    parser = argparse.ArgumentParser(
        description="Genomic Foundation Model Benchmark Suite (Single Model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required argument
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        default="RGB",
        choices=["RGB", "PGB", "GUE", "GB", "BEACON"],
        help="Path to the BEACON benchmark root directory.",
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
        "-o",
        "--overwrite",
        type=bool,
        default=False,
        help="Overwrite existing results.",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "-x",
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to use for training.",
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=5,
        help="Patience value for early stopping.",
    )
    parser.add_argument(
        "--use_hf_trainer",
        action="store_true",
        help="Use HuggingFace Trainer for training.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to the Deepspeed configuration file.",
    )
    parser.add_argument(
        "--bs_scale_factor",
        type=int,
        default=1,
        help="Batch size scale factor.",
    )
    return parser


def run_bench():
    os.makedirs("autobench_logs", exist_ok=True)
    # Generate a timestamped log filename.
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"autobench_logs/AutoBench-{time_str}.log"

    # Build the base command string.
    # This will run: accelerate launch <this_script> <other_arguments>
    cmd_base = f"accelerate launch {__file__} " + " ".join(sys.argv[1:])

    # Use platform-specific tee commands:
    if platform.system() == "Windows":
        # On Windows, use PowerShell's tee-object.
        # The command below launches PowerShell and passes the tee-object command.
        cmd = f'{cmd_base} 2>&1 | powershell -Command "tee-object -FilePath \'{log_file}\'"'
    else:
        # On Unix-like systems, use the standard tee command.
        cmd = f"{cmd_base} 2>&1 | tee {log_file}"

    # Execute the command.
    sys.exit(os.system(cmd))

if __name__ == "__main__":
    bench_command()