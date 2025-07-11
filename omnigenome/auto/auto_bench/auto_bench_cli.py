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
from omnigenome import AutoBench
from omnigenome.src.misc.utils import fprint


def bench_command(args: Optional[list] = None):
    """
    Entry point for the OmniGenome benchmark command-line interface.

    This function parses command-line arguments, initializes the AutoBench,
    and runs the evaluation.

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

    :return: An `argparse.ArgumentParser` instance.
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

    return parser


def run_bench():
    """
    Wrapper function to run the benchmark command.

    This function sets up logging, constructs the command to execute
    (potentially with `accelerate launch`), and runs it.
    """
    fprint("Running benchmark, this may take a while, please be patient...")
    fprint("You can find the logs in the 'autobench_logs' directory.")
    fprint("You can find the metrics in the 'autobench_evaluations' directory.")
    fprint(
        "If you don't intend to use accelerate, please add '--trainer native' false' to the command."
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

    # # 匹配tqdm进度条的正则表达式（根据实际输出调整）
    # tqdm_pattern = re.compile(r'^.*\d+%\|.*\|\s+\d+/\d+\s+\[.*\]\s*$')
    #
    # last_tqdm_line = ''
    #
    # with open(log_file, 'w', encoding='utf-8') as log_file:
    #     # 执行命令并捕获输出流
    #     proc = subprocess.Popen(
    #         cmd_base,
    #         shell=True,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT,
    #         bufsize=1,
    #         universal_newlines=True
    #     )
    #
    #     # 实时处理输出流
    #     for line in proc.stdout:
    #         line = line.rstrip()  # 移除行尾换行符
    #         if tqdm_pattern.match(line):
    #             # 更新最后一行tqdm输出
    #             last_tqdm_line = line + '\n'  # 换行符需要手动添加
    #             # 实时显示进度条（覆盖模式）
    #             sys.stdout.write('\r' + line)
    #             sys.stdout.flush()
    #         else:
    #             # 写入日志并正常打印
    #             log_file.write(line + '\n')
    #             print(line)
    #
    #     # 命令执行完毕后写入最后一个tqdm进度条
    #     if last_tqdm_line:
    #         log_file.write(last_tqdm_line)
    #         sys.stdout.write('\n')  # 最后换行避免覆盖
    #
    # sys.exit(proc.returncode)


if __name__ == "__main__":
    bench_command()
