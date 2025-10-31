# -*- coding: utf-8 -*-
# file: ogb_cli.py
# time: 14:00 23/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

"""
OmniGenBench (OGB) Command Line Interface

This is the main entry point for all OmniGenBench CLI commands.
It provides four main subcommands:
- autobench: Automated benchmarking of genomic foundation models
- autotrain: Automated training/fine-tuning of models
- autoinfer: Automated inference with fine-tuned models
- rna_design: RNA sequence design for target structures
"""

import argparse
import sys
import warnings
from omnigenbench import fprint

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")


def create_autoinfer_parser(subparsers):
    """Create the autoinfer subcommand parser."""
    infer_parser = subparsers.add_parser(
        "autoinfer",
        help="Run inference with fine-tuned models",
        description="Run inference with fine-tuned genomic foundation models on arbitrary sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sequence inference
  ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"
  
  # Batch inference from JSON
  ogb autoinfer --model yangheng/ogb_te_finetuned --input-file sequences.json --batch-size 64
  
  # CSV input with metadata
  ogb autoinfer --model yangheng/ogb_tfb_finetuned --input-file data.csv --device cuda:0
        """,
    )

    infer_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the fine-tuned model (e.g., yangheng/ogb_tfb_finetuned)",
    )
    infer_parser.add_argument(
        "--sequence",
        type=str,
        help="Input sequence(s). Can be a single sequence string or path to a file",
    )
    infer_parser.add_argument(
        "--input-file",
        type=str,
        help="Path to JSON/CSV file with input data",
    )
    infer_parser.add_argument(
        "--output-file",
        type=str,
        default="inference_results.json",
        help="Output file to save predictions (default: inference_results.json)",
    )
    infer_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified",
    )

    infer_parser.set_defaults(func=run_autoinfer)
    return infer_parser


def create_autotrain_parser(subparsers):
    """Create the autotrain subcommand parser."""
    train_parser = subparsers.add_parser(
        "autotrain",
        help="Automated training/fine-tuning of models",
        description="Automatically train or fine-tune genomic foundation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  ogb autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M
  
  # Training with custom parameters
  ogb autotrain --dataset ./my_dataset --model yangheng/OmniGenome-186M --num-epochs 10
        """,
    )

    train_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name or path of the dataset to train on",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the pre-trained model to fine-tune",
    )
    train_parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer to use (default: same as model)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the fine-tuned model",
    )
    train_parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    train_parser.add_argument(
        "--trainer",
        type=str,
        default="accelerate",
        help="Trainer type (default: accelerate)",
    )

    train_parser.set_defaults(func=run_autotrain)
    return train_parser


def create_autobench_parser(subparsers):
    """Create the autobench subcommand parser."""
    bench_parser = subparsers.add_parser(
        "autobench",
        help="Automated benchmarking of genomic foundation models",
        description="Automatically benchmark genomic foundation models on standard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark on RGB dataset
  ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB
  
  # Benchmark with custom trainer
  ogb autobench --model zhihan1996/DNABERT-2-117M --benchmark GUE --trainer accelerate
        """,
    )

    bench_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model name or path to benchmark",
    )
    bench_parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        required=True,
        help="Benchmark dataset name (e.g., RGB, GUE, PGB, BEACON)",
    )
    bench_parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        default=None,
        help="Tokenizer to use (default: same as model)",
    )
    bench_parser.add_argument(
        "--trainer",
        type=str,
        default="accelerate",
        help="Trainer type (default: accelerate)",
    )
    bench_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )

    bench_parser.set_defaults(func=run_autobench)
    return bench_parser


def run_autoinfer(args):
    """Execute the autoinfer command."""
    import json
    import pandas as pd
    from omnigenbench import ModelHub

    # Validate that at least one input source is provided
    if not args.sequence and not args.input_file:
        fprint(
            "Error: Either --sequence or --input-file must be provided for inference"
        )
        sys.exit(1)

    # Load the model
    fprint(f"🔄 Loading model from: {args.model}")
    model = ModelHub.load(args.model, device=args.device)
    model.eval()
    fprint(f"✅ Model loaded successfully on device: {args.device}")

    # Prepare input sequences
    sequences = []
    metadata = []

    if args.sequence:
        # Single sequence or comma-separated sequences
        if args.sequence.endswith(".txt"):
            # Read from text file (one sequence per line)
            with open(args.sequence, "r") as f:
                sequences = [line.strip() for line in f if line.strip()]
        else:
            # Direct sequence input (support comma-separated)
            sequences = [s.strip() for s in args.sequence.split(",") if s.strip()]
        metadata = [{"index": i} for i in range(len(sequences))]

    elif args.input_file:
        # Load from JSON or CSV file
        if args.input_file.endswith(".json"):
            with open(args.input_file, "r") as f:
                data = json.load(f)

            if isinstance(data, dict):
                if "sequences" in data:
                    sequences = data["sequences"]
                    metadata = [{"index": i} for i in range(len(sequences))]
                elif "data" in data:
                    # Complex format with metadata
                    for item in data["data"]:
                        sequences.append(item["sequence"])
                        meta = {k: v for k, v in item.items() if k != "sequence"}
                        metadata.append(meta)
                else:
                    raise ValueError("JSON file must contain 'sequences' or 'data' key")
            elif isinstance(data, list):
                sequences = data
                metadata = [{"index": i} for i in range(len(sequences))]

        elif args.input_file.endswith(".csv"):
            df = pd.read_csv(args.input_file)
            if "sequence" not in df.columns:
                raise ValueError("CSV file must have a 'sequence' column")
            sequences = df["sequence"].tolist()
            metadata = df.drop(columns=["sequence"]).to_dict("records")

        else:
            raise ValueError("Input file must be .json, .csv, or .txt format")

    fprint(f"📊 Processing {len(sequences)} sequence(s)...")

    # Run inference
    results = []
    for i in range(0, len(sequences), args.batch_size):
        batch_sequences = sequences[i : i + args.batch_size]
        batch_meta = metadata[i : i + args.batch_size]

        fprint(
            f"🔄 Inferring batch {i // args.batch_size + 1}/{(len(sequences) + args.batch_size - 1) // args.batch_size}..."
        )

        for seq, meta in zip(batch_sequences, batch_meta):
            try:
                output = model.inference(seq)

                # Format output based on model type
                result = {
                    "sequence": seq,
                    "metadata": meta,
                }

                # Add predictions based on output structure
                if isinstance(output, dict):
                    # Model returns dictionary with predictions/probabilities
                    if "predictions" in output:
                        result["predictions"] = (
                            output["predictions"].tolist()
                            if hasattr(output["predictions"], "tolist")
                            else output["predictions"]
                        )
                    if "probabilities" in output:
                        result["probabilities"] = (
                            output["probabilities"].tolist()
                            if hasattr(output["probabilities"], "tolist")
                            else output["probabilities"]
                        )
                    if "logits" in output:
                        result["logits"] = (
                            output["logits"].tolist()
                            if hasattr(output["logits"], "tolist")
                            else output["logits"]
                        )
                    # Include any other keys from the output
                    for key, value in output.items():
                        if key not in ["predictions", "probabilities", "logits"]:
                            result[key] = (
                                value.tolist() if hasattr(value, "tolist") else value
                            )
                else:
                    # Model returns raw tensor/array
                    result["output"] = (
                        output.tolist() if hasattr(output, "tolist") else output
                    )

                results.append(result)

            except Exception as e:
                fprint(f"⚠️  Error processing sequence {meta.get('index', i)}: {e}")
                results.append(
                    {
                        "sequence": seq,
                        "metadata": meta,
                        "error": str(e),
                    }
                )

    # Save results
    output_data = {
        "model": args.model,
        "total_sequences": len(sequences),
        "results": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    fprint(f"📁 Results saved to: {args.output_file}")
    fprint(
        f"📊 Successfully processed: {len([r for r in results if 'error' not in r])}/{len(sequences)} sequences"
    )


def run_autotrain(args):
    """Execute the autotrain command."""
    from omnigenbench.auto.auto_train.auto_train_cli import train_command

    # Convert args namespace to list format expected by train_command
    cmd_args = [
        "--dataset",
        args.dataset,
        "--model",
        args.model,
    ]

    if args.tokenizer:
        cmd_args.extend(["--tokenizer", args.tokenizer])
    if args.output_dir:
        cmd_args.extend(["--output-dir", args.output_dir])
    if args.num_epochs:
        cmd_args.extend(["--num-epochs", str(args.num_epochs)])
    if args.batch_size:
        cmd_args.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate:
        cmd_args.extend(["--learning-rate", str(args.learning_rate)])
    if args.overwrite:
        cmd_args.append("--overwrite")
    if args.trainer:
        cmd_args.extend(["--trainer", args.trainer])

    train_command(cmd_args)


def run_autobench(args):
    """Execute the autobench command."""
    from omnigenbench.auto.auto_bench.auto_bench_cli import bench_command

    # Convert args namespace to list format expected by bench_command
    cmd_args = [
        "--model_name_or_path",
        args.model,
        "--benchmark",
        args.benchmark,
    ]

    if args.tokenizer:
        cmd_args.extend(["--tokenizer", args.tokenizer])
    if args.trainer:
        cmd_args.extend(["--trainer", args.trainer])
    if args.overwrite:
        cmd_args.append("--overwrite")

    bench_command(cmd_args)


def create_rna_design_parser(subparsers):
    """Create the parser for RNA design command."""
    parser = subparsers.add_parser(
        "rna_design",
        help="Design RNA sequences for target secondary structures",
        description="""
Design RNA sequences that fold into specified secondary structures using
a genetic algorithm guided by masked language modeling.

The algorithm uses:
- ViennaRNA for structure prediction and free energy calculation
- Multi-objective optimization (structure similarity + energy stability)
- MLM-guided mutations for biologically plausible sequences

Examples:
  # Simple hairpin design
  ogb rna_design --structure "(((...)))"
  
  # Stem-loop with specific model
  ogb rna_design --structure "(((((.....)))))(((...)))" --model yangheng/OmniGenome-186M
  
  # High mutation rate for diverse exploration
  ogb rna_design --structure "((((....))))((((...))))" --mutation-ratio 0.3
  
  # Save results to file
  ogb rna_design --structure "(((...)))" --output-file designs.txt
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--structure",
        type=str,
        required=True,
        help="Target RNA secondary structure in dot-bracket notation. "
        "Use '(' for open base pairs, ')' for closing pairs, and '.' for unpaired bases. "
        "Example: '(((...)))' represents a simple hairpin structure.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yangheng/OmniGenome-186M",
        help="Pre-trained model name or path for MLM-guided mutations. "
        "Default: yangheng/OmniGenome-186M. Use larger models like OmniGenome-418M "
        "for better biological plausibility.",
    )

    parser.add_argument(
        "--mutation-ratio",
        type=float,
        default=0.1,
        help="Fraction of nucleotides to mutate in each generation (0.0-1.0). "
        "Default: 0.1 (10%%). Lower values (0.05) for conservative exploration, "
        "higher values (0.2-0.3) for diverse exploration. Higher ratios may reduce "
        "convergence speed.",
    )

    parser.add_argument(
        "--num-population",
        type=int,
        default=100,
        help="Population size for genetic algorithm. Default: 100. "
        "Larger populations (200-500) explore more diversity but take longer. "
        "Smaller populations (50) converge faster but may miss optimal solutions.",
    )

    parser.add_argument(
        "--num-generation",
        type=int,
        default=100,
        help="Maximum number of generations. Default: 100. "
        "The algorithm terminates early if a perfect match is found. "
        "Increase to 200-500 for complex structures or if no perfect match is found.",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path to save designed sequences. "
        "If not specified, sequences are printed to stdout. "
        "Each line contains: sequence, predicted structure, normalized distance, free energy.",
    )

    parser.set_defaults(func=run_rna_design)
    return parser


def run_rna_design(args):
    """Execute the RNA design command."""
    from omnigenbench.cli.commands.rna.rna_design import RNADesignCommand

    # Execute the command with args namespace
    RNADesignCommand.execute(args)


def main():
    """
    Main entry point for the OGB (OmniGenBench) CLI.

    This provides a unified interface for all OmniGenBench command-line tools.
    """
    parser = argparse.ArgumentParser(
        prog="ogb",
        description="OmniGenBench (OGB) - Unified CLI for Genomic Foundation Model Development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  autoinfer   - Run inference with fine-tuned models
  autotrain   - Train or fine-tune genomic models
  autobench   - Benchmark models on standard datasets
  rna_design  - Design RNA sequences for target secondary structures

Examples:
  ogb autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"
  ogb autotrain --dataset yangheng/tfb_promoters --model zhihan1996/DNABERT-2-117M
  ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB
  ogb rna_design --structure "(((...)))" --model yangheng/OmniGenome-186M

For more information: https://github.com/yangheng95/OmniGenBench
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="OmniGenBench v0.3.23alpha",
    )

    # Create subparsers for each command
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available OmniGenBench commands",
        dest="command",
        required=True,
        help="Command to execute",
    )

    # Add subcommands
    create_autoinfer_parser(subparsers)
    create_autotrain_parser(subparsers)
    create_autobench_parser(subparsers)
    create_rna_design_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Execute the selected command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
