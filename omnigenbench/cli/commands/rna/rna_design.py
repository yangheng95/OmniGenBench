# -*- coding: utf-8 -*-
# file: rna_design.py
# time: 19:06 05/02/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.
import json
import argparse
from omnigenbench import OmniModelForRNADesign
from ..base import BaseCommand


class RNADesignCommand(BaseCommand):
    """
    This class provides a CLI interface for designing RNA sequences that fold into
    specific secondary structures. It uses genetic algorithms with customizable
    parameters to optimize sequence design for target structures.

    The design process involves:

    1. Loading a pre-trained RNA design model
    2. Running genetic algorithm optimization
    3. Generating sequences that match the target structure
    4. Saving results to file (optional)

    Attributes:
        model (str): Path or name of the pre-trained RNA design model
        structure (str): Target RNA secondary structure in dot-bracket notation
        mutation_ratio (float): Genetic algorithm mutation rate
        num_population (int): Population size for genetic algorithm
        num_generation (int): Number of generations for evolution

    Example:
        >>> # Basic RNA design
        >>> python -m omnigenbench.cli.omnigenome_cli rna_design --structure "(((...)))"
        >>> # Design with custom parameters
        >>> python -m omnigenbench.cli.omnigenome_cli rna_design \\
        ...     --structure "(((...)))" \\
        ...     --model "yangheng/OmniGenome-186M" \\
        ...     --mutation-ratio 0.3 \\
        ...     --num-population 200 \\
        ...     --num-generation 150 \\
        ...     --output-file "results.json"
    """

    @classmethod
    def register_command(cls, subparsers):
        """
        This method sets up the command-line interface for RNA sequence design,
        including all necessary arguments and their descriptions.

        Args:
            subparsers: The subparsers object from argparse to add the command to

        Example:
            >>> parser = argparse.ArgumentParser()
            >>> subparsers = parser.add_subparsers()
            >>> RNADesignCommand.register_command(subparsers)
        """
        parser: argparse.ArgumentParser = subparsers.add_parser(
            "rna_design",
            help="RNA Sequence Design based on Secondary Structure, Using Genetic Algorithm by OmniGenome",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--structure",
            required=True,
            help="The target RNA structure in dot-bracket notation (e.g., '(((...)))')",
        )
        parser.add_argument(
            "--model",
            default="yangheng/OmniGenome-186M",
            help="Model name or path to the pre-trained model (default: yangheng/OmniGenome-186M)",
        )
        parser.add_argument(
            "--mutation-ratio",
            type=float,
            default=0.5,
            help="Mutation ratio for genetic algorithm (0.0-1.0, default: 0.5)",
        )
        parser.add_argument(
            "--num-population",
            type=int,
            default=100,
            help="Number of individuals in population (default: 100)",
        )
        parser.add_argument(
            "--num-generation",
            type=int,
            default=100,
            help="Number of generations to evolve (default: 100)",
        )
        parser.add_argument(
            "--output-file", type=str, help="Output JSON file to save results"
        )
        cls.add_common_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @staticmethod
    def execute(args: argparse.Namespace):
        """
        This method runs the RNA sequence design process using genetic algorithms.
        It validates parameters, loads the model, runs the design optimization,
        and outputs or saves the results.

        Args:
            args (argparse.Namespace): Parsed command-line arguments containing
                                      design parameters and model settings

        Raises:
            ValueError: If mutation_ratio is not between 0.0 and 1.0

        Example:
            >>> args = parser.parse_args(['design', '--structure', '(((...)))'])
            >>> RNADesignCommand.execute(args)
        """
        # 参数验证逻辑
        if not 0 <= args.mutation_ratio <= 1:
            raise ValueError("--mutation-ratio should be between 0.0 and 1.0")

        # 核心业务逻辑
        model = OmniModelForRNADesign(model=args.model)
        best_sequences = model.design(
            structure=args.structure,
            mutation_ratio=args.mutation_ratio,
            num_population=args.num_population,
            num_generation=args.num_generation,
        )

        # 结果输出
        print(f"The best RNA sequences for {args.structure}:")
        if isinstance(best_sequences, list):
            for seq in best_sequences:
                print(f"- {seq}")
        else:
            # Fallback for single sequence (shouldn't happen with updated model)
            print(f"- {best_sequences}")
            best_sequences = [best_sequences]

        # 结果保存
        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(
                    {
                        "structure": args.structure,
                        "parameters": {
                            "model": args.model,
                            "mutation_ratio": args.mutation_ratio,
                            "num_population": args.num_population,
                            "num_generation": args.num_generation,
                        },
                        "best_sequences": best_sequences,
                    },
                    f,
                    indent=2,
                )
            print(f"\nResults saved to {args.output_file}")


def register_command(subparsers):
    """
    This function is a convenience wrapper for registering the RNADesignCommand
    with the argument parser.

    Args:
        subparsers: The subparsers object from argparse to add the command to

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> subparsers = parser.add_subparsers()
        >>> register_command(subparsers)
    """
    RNADesignCommand.register_command(subparsers)
