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
    from ..auto_bench.auto_bench import AutoBench
    from ...src.misc.utils import fprint
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from omnigenbench.auto.auto_bench.auto_bench import AutoBench
    from omnigenbench.src.misc.utils import fprint


def bench_command(args: Optional[list] = None):
    """
    This function parses command-line arguments, initializes the AutoBench,
    and runs the evaluation.
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

    # Initialize benchmark
    autobench = AutoBench(
        benchmark=parsed_args.benchmark,
        model_name_or_path=model,
        tokenizer=tokenizer,
        overwrite=parsed_args.overwrite,
        trainer=parsed_args.trainer,
    )

    # Run evaluation
    autobench.run(**vars(parsed_args))


def create_parser() -> argparse.ArgumentParser:
    """
    Creates the argument parser for the benchmark CLI.

    Returns:
        An `argparse.ArgumentParser` instance.
    """
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
        # choices=["RGB", "PGB", "GUE", "GB", "BEACON"],
        help="Path to the BEACON benchmark root directory.",
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
        help="Overwrite existing bench results, otherwise resume from benchmark checkpoint.",
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
    parser.add_argument(
        "--lora",
        default=False,
        type=bool,
        help="Use LoRA fine-tuning if this flag is set.",
    )
    return parser


def run_bench():
    """
    This function sets up logging, constructs the command to execute
    (potentially with `accelerate launch`), and runs it.
    """
    fprint("Running benchmark, this may take a while, please be patient...")
    fprint("You can find the logs in the 'autobench_logs' directory.")
    fprint("You can find the metrics in the 'autobench_evaluations' directory.")
    fprint(
        "If you don't intend to use accelerate, please add '--trainer native' to the command."
    )
    fprint(
        "If you want to alter accelerate's behavior, please refer to 'accelerate config' command."
    )
    fprint("If you encounter any issues, please report them on the GitHub repository.")
    os.makedirs("autobench_logs", exist_ok=True)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
    log_file = f"autobench_logs/AutoBench-{time_str}.log"
    from pathlib import Path

    try:
        mixed_precision = sys.argv[sys.argv.index("--autocast") + 1].lower()
    except ValueError:
        mixed_precision = "fp16"
    file_path = Path(__file__).resolve()
    if (
        "--trainer" in sys.argv
        and sys.argv[sys.argv.index("--trainer") + 1].lower() == "native"
    ):
        cmd_base = f'python "{file_path}" ' + " ".join(sys.argv[1:])
    else:
        cmd_base = (
            f'accelerate launch --mixed_precision "{mixed_precision}" "{file_path}" '
            + " ".join(sys.argv[1:])
        )

    # Use platform-specific tee commands:
    if platform.system() == "Windows":
        # On Windows, use PowerShell's tee-object.
        # The command below launches PowerShell and passes the tee-object command.
        # try:
        #     cmd = f"{cmd_base} 2>&1 | powershell -Command Get-Content {log_file} -Wait"
        # except Exception as e:
        #     fprint(f"The log file cannot be saved due to Error: {e}")
        #     fprint(
        #         "If commands not allowed in PowerShell, "
        #         "please run 'Set-ExecutionPolicy RemoteSigned' in PowerShell with Admin."
        #     )
        cmd = f"{cmd_base} 2>&1"
    else:
        # On Unix-like systems, use the standard tee command.
        cmd = f"{cmd_base} 2>&1 | tee '{log_file}'"

    # Execute the command.
    sys.exit(os.system(cmd))


if __name__ == "__main__":
    bench_command()
