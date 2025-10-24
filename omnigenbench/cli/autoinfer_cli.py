# -*- coding: utf-8 -*-
# file: autoinfer_cli.py
# time: 12:00 23/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import argparse
import json
import warnings

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")


def main():
    """
    Entry point for the autoinfer command-line interface.

    This function provides a standalone CLI for running inference with fine-tuned
    genomic foundation models on arbitrary sequences.

    Example:
        >>> # Single sequence inference
        >>> autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"
        >>>
        >>> # Batch inference from file
        >>> autoinfer --model yangheng/ogb_te_finetuned --input-file sequences.json
    """
    parser = argparse.ArgumentParser(
        description="OmniGenBench AutoInfer - Run inference with fine-tuned genomic models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sequence inference
  autoinfer --model yangheng/ogb_tfb_finetuned --sequence "ATCGATCGATCG"
  
  # Batch inference from JSON
  autoinfer --model yangheng/ogb_te_finetuned --input-file sequences.json --batch-size 64
  
  # CSV input with metadata
  autoinfer --model yangheng/ogb_tfb_finetuned --input-file data.csv --device cuda:0
  
For more information, visit: https://github.com/yangheng95/OmniGenBench
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or name of the fine-tuned model (e.g., yangheng/ogb_tfb_finetuned)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Input sequence(s). Can be a single sequence string or path to a file",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to JSON/CSV file with input data. JSON format: {'sequences': [...]} or {'data': [{'sequence': ..., ...}]}",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="inference_results.json",
        help="Output file to save predictions (default: inference_results.json)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (e.g., 'cuda:0', 'cpu'). Auto-detected if not specified",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup
    import pandas as pd
    from omnigenbench import ModelHub

    # Validate that at least one input source is provided
    if not args.sequence and not args.input_file:
        parser.error("Either --sequence or --input-file must be provided for inference")

    # Load the model
    print(f"🔄 Loading model from: {args.model}")
    model = ModelHub.load(args.model, device=args.device)
    model.eval()
    print(f"✅ Model loaded successfully on device: {args.device}")

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

    print(f"📊 Processing {len(sequences)} sequence(s)...")

    # Run inference
    results = []
    for i in range(0, len(sequences), args.batch_size):
        batch_sequences = sequences[i : i + args.batch_size]
        batch_meta = metadata[i : i + args.batch_size]

        print(
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
                print(f"⚠️  Error processing sequence {meta.get('index', i)}: {e}")
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

    print(f"✅ Inference completed!")
    print(f"📁 Results saved to: {args.output_file}")
    print(
        f"📊 Successfully processed: {len([r for r in results if 'error' not in r])}/{len(sequences)} sequences"
    )


if __name__ == "__main__":
    main()
