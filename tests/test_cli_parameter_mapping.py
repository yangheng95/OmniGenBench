# -*- coding: utf-8 -*-
# file: test_cli_parameter_mapping.py
# time: 09:58 08/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test CLI parameter mapping between ogb_cli.py and underlying command implementations.
Ensures that parameter names match correctly to avoid parsing errors.
"""

import pytest
import argparse
import inspect
from omnigenbench.cli import ogb_cli
from omnigenbench.auto.auto_bench.auto_bench_cli import create_parser as create_bench_parser
from omnigenbench.auto.auto_train.auto_train_cli import create_parser as create_train_parser


class TestCLIParameterMapping:
    """Test that CLI parameters are correctly mapped between ogb and underlying commands"""

    def test_autobench_parameter_mapping(self):
        """Test autobench parameter mapping from ogb to bench_command"""
        # Get ogb autobench parser
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ogb_cli.create_autobench_parser(subparsers)
        
        # Parse test arguments
        test_args = parser.parse_args([
            'autobench',
            '--model', 'test_model',
            '--benchmark', 'RGB',
            '--tokenizer', 'test_tokenizer',
            '--trainer', 'accelerate',
            '--overwrite'
        ])
        
        # Verify all attributes exist
        assert hasattr(test_args, 'model')
        assert hasattr(test_args, 'benchmark')
        assert hasattr(test_args, 'tokenizer')
        assert hasattr(test_args, 'trainer')
        assert hasattr(test_args, 'overwrite')
        
        # Get bench_command parser
        bench_parser = create_bench_parser()
        
        # Test that run_autobench creates correct cmd_args format
        # Simulate what run_autobench does
        cmd_args = [
            "--model", test_args.model,
            "--benchmark", test_args.benchmark,
        ]
        
        if test_args.tokenizer:
            cmd_args.extend(["--tokenizer", test_args.tokenizer])
        if test_args.trainer:
            cmd_args.extend(["--trainer", test_args.trainer])
        if test_args.overwrite:
            cmd_args.append("--overwrite")
        
        # Verify bench_parser can parse these args
        parsed_bench_args = bench_parser.parse_args(cmd_args)
        assert parsed_bench_args.model == 'test_model'
        assert parsed_bench_args.benchmark == 'RGB'
        assert parsed_bench_args.tokenizer == 'test_tokenizer'
        assert parsed_bench_args.trainer == 'accelerate'

    def test_autotrain_parameter_mapping(self):
        """Test autotrain parameter mapping from ogb to train_command"""
        # Get ogb autotrain parser
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ogb_cli.create_autotrain_parser(subparsers)
        
        # Parse test arguments with all optional parameters
        test_args = parser.parse_args([
            'autotrain',
            '--dataset', 'test_dataset',
            '--model', 'test_model',
            '--tokenizer', 'test_tokenizer',
            '--output-dir', './output',
            '--num-epochs', '10',
            '--batch-size', '8',
            '--learning-rate', '2e-5',
            '--trainer', 'accelerate',
            '--overwrite'
        ])
        
        # Verify all attributes exist with correct names
        assert hasattr(test_args, 'dataset')
        assert hasattr(test_args, 'model')
        assert hasattr(test_args, 'tokenizer')
        assert hasattr(test_args, 'output_dir')  # Note: argparse converts - to _
        assert hasattr(test_args, 'num_epochs')
        assert hasattr(test_args, 'batch_size')
        assert hasattr(test_args, 'learning_rate')
        assert hasattr(test_args, 'trainer')
        assert hasattr(test_args, 'overwrite')
        
        # Get train_command parser
        train_parser = create_train_parser()
        
        # Simulate what run_autotrain does
        cmd_args = [
            "--dataset", test_args.dataset,
            "--model", test_args.model,
        ]
        
        if test_args.tokenizer:
            cmd_args.extend(["--tokenizer", test_args.tokenizer])
        if test_args.output_dir:
            cmd_args.extend(["--output-dir", test_args.output_dir])
        if test_args.num_epochs:
            cmd_args.extend(["--num-epochs", str(test_args.num_epochs)])
        if test_args.batch_size:
            cmd_args.extend(["--batch-size", str(test_args.batch_size)])
        if test_args.learning_rate:
            cmd_args.extend(["--learning-rate", str(test_args.learning_rate)])
        if test_args.trainer:
            cmd_args.extend(["--trainer", test_args.trainer])
        if test_args.overwrite:
            cmd_args.append("--overwrite")
        
        # Verify train_parser can parse these args
        parsed_train_args, _ = train_parser.parse_known_args(cmd_args)
        assert parsed_train_args.dataset == 'test_dataset'
        assert parsed_train_args.model == 'test_model'
        assert parsed_train_args.tokenizer == 'test_tokenizer'

    def test_autobench_short_options(self):
        """Test that short options (-m, -b, -t) work correctly"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ogb_cli.create_autobench_parser(subparsers)
        
        # Test with short options
        test_args = parser.parse_args([
            'autobench',
            '-m', 'test_model',
            '-b', 'RGB',
            '-t', 'test_tokenizer',
        ])
        
        assert test_args.model == 'test_model'
        assert test_args.benchmark == 'RGB'
        assert test_args.tokenizer == 'test_tokenizer'

    def test_autotrain_parameter_name_consistency(self):
        """Test that parameter names in ogb_cli match those expected by train_command"""
        # Check run_autotrain function
        import inspect
        run_autotrain_source = inspect.getsource(ogb_cli.run_autotrain)
        
        # Verify correct parameter names are used
        assert '--dataset' in run_autotrain_source
        assert '--model' in run_autotrain_source
        assert '--tokenizer' in run_autotrain_source
        assert '--output-dir' in run_autotrain_source
        assert '--num-epochs' in run_autotrain_source
        assert '--batch-size' in run_autotrain_source
        assert '--learning-rate' in run_autotrain_source
        assert '--trainer' in run_autotrain_source
        
        # Make sure wrong names are NOT used
        assert '--config_or_model' not in run_autotrain_source
        assert '--config-or-model' not in run_autotrain_source

    def test_autobench_parameter_name_consistency(self):
        """Test that parameter names in ogb_cli match those expected by bench_command"""
        import inspect
        run_autobench_source = inspect.getsource(ogb_cli.run_autobench)
        
        # Verify correct parameter names are used
        assert '--model' in run_autobench_source
        assert '--benchmark' in run_autobench_source
        assert '--tokenizer' in run_autobench_source
        assert '--trainer' in run_autobench_source
        
        # Make sure wrong names are NOT used (the bug we just fixed)
        assert '--config_or_model' not in run_autobench_source
        assert '--config-or-model' not in run_autobench_source

    def test_rna_design_parameters(self):
        """Test RNA design parameter parsing"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ogb_cli.create_rna_design_parser(subparsers)
        
        # Parse with all parameters
        test_args = parser.parse_args([
            'rna_design',
            '--structure', '(((...)))',
            '--model', 'yangheng/OmniGenome-186M',
            '--mutation-ratio', '0.15',
            '--num-population', '200',
            '--num-generation', '150',
            '--output-file', 'output.txt'
        ])
        
        assert test_args.structure == '(((...)))'
        assert test_args.model == 'yangheng/OmniGenome-186M'
        assert test_args.mutation_ratio == 0.15
        assert test_args.num_population == 200
        assert test_args.num_generation == 150
        assert test_args.output_file == 'output.txt'

    def test_autoinfer_parameters(self):
        """Test autoinfer parameter parsing"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ogb_cli.create_autoinfer_parser(subparsers)
        
        # Test with sequence
        test_args = parser.parse_args([
            'autoinfer',
            '--model', 'test_model',
            '--sequence', 'ATCGATCG',
            '--batch-size', '16',
            '--device', 'cuda:0',
            '--output-file', 'results.json'
        ])
        
        assert test_args.model == 'test_model'
        assert test_args.sequence == 'ATCGATCG'
        assert test_args.batch_size == 16
        assert test_args.device == 'cuda:0'
        assert test_args.output_file == 'results.json'


class TestCLIArgumentConsistency:
    """Test consistency of argument handling across all CLI commands"""
    
    def test_all_commands_have_help(self):
        """Verify all subcommands provide help"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        ogb_cli.create_autoinfer_parser(subparsers)
        ogb_cli.create_autotrain_parser(subparsers)
        ogb_cli.create_autobench_parser(subparsers)
        ogb_cli.create_rna_design_parser(subparsers)
        
        # Each command should have a help attribute
        for action in subparsers._choices_actions:
            assert action.help is not None

    def test_boolean_flags_consistency(self):
        """Test that boolean flags use action='store_true' consistently"""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        # Check autobench
        bench_parser = ogb_cli.create_autobench_parser(subparsers)
        bench_actions = {action.dest: action for action in bench_parser._actions}
        if 'overwrite' in bench_actions:
            # Should use store_true, not type=bool
            # For _StoreTrueAction, check the class name instead of .action attribute
            action_type = type(bench_actions['overwrite']).__name__
            assert action_type == '_StoreTrueAction', f"Expected _StoreTrueAction, got {action_type}"
        
        # Check autotrain
        train_parser = ogb_cli.create_autotrain_parser(subparsers)
        train_actions = {action.dest: action for action in train_parser._actions}
        if 'overwrite' in train_actions:
            action_type = type(train_actions['overwrite']).__name__
            assert action_type == '_StoreTrueAction', f"Expected _StoreTrueAction, got {action_type}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
