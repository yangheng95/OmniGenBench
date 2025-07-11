# -*- coding: utf-8 -*-
# file: auto_bench_cli.py
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
import sys
import time
from pathlib import Path

from ....auto.auto_bench.auto_bench import AutoBench
from ....src.misc.utils import fprint
from ..base import BaseCommand


class BenchCommand(BaseCommand):
    """
    Command-line interface for running automated benchmarking of genomic foundation models.
    
    This class provides a CLI interface for the AutoBench functionality, allowing users
    to easily run comprehensive evaluations of genomic models across multiple benchmarks.
    It supports various benchmarks, models, and training configurations.
    
    Attributes:
        benchmarks (list): List of available benchmarks (RGB, PGB, GUE, GB, BEACON)
        trainers (list): List of available trainers (native, accelerate, hf_trainer)
        
    Example:
        >>> # Run basic benchmark
        >>> python -m omnigenome.cli autobench --model "model_name" --benchmark "RGB"
        
        >>> # Run with custom settings
        >>> python -m omnigenome.cli autobench \
        ...     --model "model_name" \
        ...     --benchmark "RGB" \
        ...     --trainer "accelerate" \
        ...     --bs_scale 2 \
        ...     --overwrite True
    """
    
    @classmethod
    def register_command(cls, subparsers):
        """
        Register the autobench command with the argument parser.
        
        This method sets up the command-line interface for the autobench functionality,
        including all necessary arguments and their descriptions.
        
        Args:
            subparsers: The subparsers object from argparse to add the command to
            
        Example:
            >>> parser = argparse.ArgumentParser()
            >>> subparsers = parser.add_subparsers()
            >>> BenchCommand.register_command(subparsers)
        """
        parser = subparsers.add_parser(
            "autobench",
            help="Run Auto-benchmarking for Genomic Foundation Models.",
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

        cls.add_common_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @staticmethod
    def execute(args: argparse.Namespace):
        """
        Execute the autobench command with the provided arguments.
        
        This method runs the automated benchmarking process using the AutoBench
        class. It handles model and tokenizer loading, benchmark execution,
        and result logging.
        
        Args:
            args (argparse.Namespace): Parsed command-line arguments containing
                                      benchmark configuration and model settings
                                      
        Example:
            >>> args = parser.parse_args(['autobench', '--model', 'model_name'])
            >>> BenchCommand.execute(args)
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
        fprint(
            "If you encounter any issues, please report them on the GitHub repository."
        )
        # 特殊模型处理
        if "multimolecule" in args.model:
            from multimolecule import RnaTokenizer, AutoModelForTokenPrediction

            tokenizer = RnaTokenizer.from_pretrained(args.model)
            model = AutoModelForTokenPrediction.from_pretrained(
                args.model, trust_remote_code=True
            ).base_model
        else:
            tokenizer = args.tokenizer
            model = args.model

        autobench = AutoBench(
            benchmark=args.benchmark,
            model_name_or_path=model,
            tokenizer=tokenizer,
            overwrite=args.overwrite,
            trainer=args.trainer,
        )
        autobench.run(**vars(args))
        log_dir = Path(args.output_dir) / "autobench_evaluations"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_file = log_dir / f"bench_{args.benchmark}_{timestamp}.log"

        cmd_base = f"{sys.executable} -m omnigenome_cli.bench_internal " + " ".join(
            f"--{k}={v}" if v is not None else f"--{k}"
            for k, v in vars(args).items()
            if k not in {"func", "output_dir", "log_level"}
        )

        if platform.system() == "Windows":
            return f"{cmd_base} 2>&1 | powershell -Command \"tee-object -FilePath '{log_file}'\""
        os.system(f"{cmd_base} 2>&1 | tee {log_file}")


def register_command(subparsers):
    """
    Register the autobench command with the CLI.
    
    This function is a convenience wrapper for registering the BenchCommand
    with the argument parser.
    
    Args:
        subparsers: The subparsers object from argparse to add the command to
        
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> subparsers = parser.add_subparsers()
        >>> register_command(subparsers)
    """
    BenchCommand.register_command(subparsers)
