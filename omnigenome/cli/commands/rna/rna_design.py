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
from pathlib import Path
from typing import Optional
from omnigenome import OmniGenomeModelForRNADesign
from ..base import BaseCommand


class RNADesignCommand(BaseCommand):
    @classmethod
    def register_command(cls, subparsers):
        parser: argparse.ArgumentParser = subparsers.add_parser(
            "design",
            help="RNA Sequence Design based on Secondary Structure, Using Genetic Algorithm by OmniGenome",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--structure",
            required=True,
            help="The target RNA structure in dot-bracket notation (e.g., '(((...)))')",
        )
        parser.add_argument(
            "--model-path",
            default="yangheng/OmniGenome-186M",
            help="Model path to the pre-trained model (default: yangheng/OmniGenome-186M)",
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
            "--output", type=Path, help="Output JSON file to save results"
        )
        cls.add_common_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @staticmethod
    def execute(args: argparse.Namespace):
        """执行设计命令"""
        # 参数验证逻辑
        if not 0 <= args.mutation_ratio <= 1:
            raise ValueError("--mutation-ratio should be between 0.0 and 1.0")

        # 核心业务逻辑
        model = OmniGenomeModelForRNADesign(model_path=args.model_path)
        best_sequences = model.design(
            structure=args.structure,
            mutation_ratio=args.mutation_ratio,
            num_population=args.num_population,
            num_generation=args.num_generation,
        )

        # 结果输出
        print(f"The best RNA sequences for {args.structure}:")
        for seq in best_sequences:
            print(f"- {seq}")

        # 结果保存
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "structure": args.structure,
                        "parameters": vars(args),
                        "best_sequences": best_sequences,
                    },
                    f,
                    indent=2,
                )
            print(f"\nResults saved to {args.output}")


def register_command(subparsers):
    RNADesignCommand.register_command(subparsers)
