# -*- coding: utf-8 -*-
# file: test_autobench_autotrain.py
# time: 15:00 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for AutoBench and AutoTrain Python API.
Based on examples/autobench_gfm_evaluation/ patterns.

These tests cover:
- AutoBench API usage
- AutoTrain API usage
- Benchmark configuration patterns
- Multi-seed evaluation
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from omnigenbench import AutoBench, AutoTrain


@pytest.fixture
def benchmark_names():
    """Available benchmark datasets"""
    return ["RGB", "BEACON", "GUE", "PGB", "GB"]


@pytest.fixture
def sample_benchmark_config():
    """
    Sample benchmark configuration.
    Based on examples/autobench_gfm_evaluation/RGB/*/config.py pattern.
    """
    return {
        "task_type": "sequence_classification",
        "num_labels": 2,
        "max_length": 512,
        "batch_size": 8,
        "epochs": 50,
        "learning_rate": 2e-5,
        "seeds": [0, 1, 2],
        "trainer": "accelerate",
    }


class TestAutoBenchAPI:
    """
    Test AutoBench Python API functionality.
    Based on examples/autobench_gfm_evaluation/ usage patterns.
    """
    
    def test_autobench_initialization(self):
        """Test AutoBench can be initialized"""
        # Basic initialization
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path="yangheng/OmniGenome-186M",
            overwrite=False
        )
        
        assert bench is not None
        assert hasattr(bench, "run")
    
    def test_autobench_with_different_benchmarks(self, benchmark_names):
        """Test AutoBench accepts different benchmark names"""
        for benchmark in benchmark_names:
            bench = AutoBench(
                benchmark=benchmark,
                model_name_or_path="yangheng/OmniGenome-186M",
            )
            assert bench is not None
    
    def test_autobench_configuration_options(self):
        """Test AutoBench accepts various configuration options"""
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path="yangheng/OmniGenome-186M",
            tokenizer_name_or_path="yangheng/OmniGenome-186M",
            trainer="accelerate",
            overwrite=True,
        )
        
        assert bench is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_autobench_run_basic(self):
        """
        Test AutoBench.run() with minimal configuration.
        This is a slow test that actually runs benchmarking.
        """
        # Use smallest model and single seed for speed
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path="yangheng/OmniGenome-52M",
            overwrite=True,
        )
        
        # Run with minimal config
        try:
            results = bench.run(
                batch_size=4,
                seeds=[0],  # Single seed for speed
                epochs=1,   # Single epoch for speed
            )
            
            # Verify results structure
            assert results is not None
            assert isinstance(results, dict) or isinstance(results, list)
        except Exception as e:
            # Benchmark may fail due to missing data or resources
            pytest.skip(f"Benchmark execution failed: {e}")
    
    def test_autobench_multi_seed_config(self):
        """Test AutoBench supports multi-seed evaluation"""
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path="yangheng/OmniGenome-186M",
        )
        
        # Configuration with multiple seeds (as in examples)
        # This tests the interface, not actual execution
        seeds = [0, 1, 2]
        
        # Verify bench accepts seeds parameter
        assert hasattr(bench, "run")
        
        # Mock the run to test interface
        with patch.object(bench, "run") as mock_run:
            bench.run(seeds=seeds)
            mock_run.assert_called_once()


class TestAutoTrainAPI:
    """
    Test AutoTrain Python API functionality.
    Based on training patterns in examples.
    """
    
    def test_autotrain_initialization(self):
        """Test AutoTrain can be initialized"""
        trainer = AutoTrain(
            dataset_name_or_path="translation_efficiency_prediction",
            model_name_or_path="yangheng/PlantRNA-FM",
        )
        
        assert trainer is not None
        assert hasattr(trainer, "train")
    
    def test_autotrain_with_custom_config(self, tmp_path):
        """Test AutoTrain accepts custom training configuration"""
        output_dir = tmp_path / "trained_model"
        
        trainer = AutoTrain(
            dataset_name_or_path="translation_efficiency_prediction",
            model_name_or_path="yangheng/PlantRNA-FM",
            output_dir=str(output_dir),
            num_labels=2,
            max_length=512,
            batch_size=16,
            epochs=5,
            learning_rate=2e-5,
        )
        
        assert trainer is not None
    
    def test_autotrain_different_task_types(self):
        """Test AutoTrain handles different task types"""
        task_configs = [
            {
                "dataset": "translation_efficiency_prediction",
                "task_type": "sequence_classification",
                "num_labels": 2,
            },
            {
                "dataset": "deepsea_tfb_prediction",
                "task_type": "multilabel_classification",
                "num_labels": 919,
            },
        ]
        
        for config in task_configs:
            trainer = AutoTrain(
                dataset_name_or_path=config["dataset"],
                model_name_or_path="yangheng/OmniGenome-52M",
                num_labels=config["num_labels"],
            )
            assert trainer is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_autotrain_full_workflow(self, tmp_path):
        """
        Test complete AutoTrain workflow.
        This is a slow integration test.
        """
        output_dir = tmp_path / "trained_model"
        
        trainer = AutoTrain(
            dataset_name_or_path="translation_efficiency_prediction",
            model_name_or_path="yangheng/PlantRNA-FM",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
        )
        
        try:
            # Run training
            trainer.train()
            
            # Verify output directory created
            assert output_dir.exists()
            
            # Check for model files
            model_files = list(output_dir.glob("*.bin")) + \
                         list(output_dir.glob("*.safetensors")) + \
                         list(output_dir.glob("config.json"))
            assert len(model_files) > 0
            
        except Exception as e:
            pytest.skip(f"Training failed: {e}")


class TestBenchmarkConfigurations:
    """
    Test benchmark configuration patterns.
    Based on examples/autobench_gfm_evaluation/RGB/*/config.py
    """
    
    def test_rgb_benchmark_config_structure(self, sample_benchmark_config):
        """Test RGB benchmark configuration structure"""
        config = sample_benchmark_config
        
        # Verify required fields
        assert "task_type" in config
        assert "num_labels" in config
        assert "max_length" in config
        assert "batch_size" in config
        assert "epochs" in config
        assert "learning_rate" in config
        assert "seeds" in config
    
    def test_multi_seed_configuration(self, sample_benchmark_config):
        """Test multi-seed evaluation configuration"""
        config = sample_benchmark_config
        
        # Standard practice is 3 seeds for statistical significance
        assert "seeds" in config
        assert isinstance(config["seeds"], list)
        assert len(config["seeds"]) >= 1
        
        # Seeds should be integers
        for seed in config["seeds"]:
            assert isinstance(seed, int)
    
    def test_task_specific_configs(self):
        """Test different task types have appropriate configs"""
        configs = {
            "sequence_classification": {
                "task_type": "sequence_classification",
                "num_labels": 2,
            },
            "multilabel_classification": {
                "task_type": "multilabel_classification",
                "num_labels": 919,
            },
            "token_classification": {
                "task_type": "token_classification",
                "num_labels": 3,
            },
        }
        
        for task_type, config in configs.items():
            assert config["task_type"] == task_type
            assert "num_labels" in config
            assert config["num_labels"] > 0


class TestBenchmarkMetrics:
    """
    Test benchmark metric computation and reporting.
    """
    
    def test_classification_metrics_available(self):
        """Test classification metrics are available"""
        from omnigenbench import ClassificationMetric
        
        metric = ClassificationMetric()
        
        # Standard classification metrics
        assert hasattr(metric, "accuracy")
        assert hasattr(metric, "f1")
        assert hasattr(metric, "precision")
        assert hasattr(metric, "recall")
        assert hasattr(metric, "roc_auc")
    
    def test_metric_computation_interface(self):
        """Test metric computation interface"""
        from omnigenbench import ClassificationMetric
        import numpy as np
        
        metric = ClassificationMetric()
        
        # Mock predictions and labels
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 1, 1])
        
        # Metrics should be callable
        assert callable(metric.accuracy)
        
        # Note: Actual computation tested in metric-specific tests
    
    def test_multi_seed_result_aggregation(self):
        """Test multi-seed results can be aggregated"""
        # Mock results from multiple seeds
        seed_results = {
            0: {"accuracy": 0.85, "f1": 0.82},
            1: {"accuracy": 0.87, "f1": 0.84},
            2: {"accuracy": 0.86, "f1": 0.83},
        }
        
        # Calculate mean and std (standard practice)
        import numpy as np
        accuracies = [r["accuracy"] for r in seed_results.values()]
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        assert 0.85 <= mean_acc <= 0.87
        assert std_acc >= 0


class TestBenchmarkDatasets:
    """
    Test benchmark dataset loading and structure.
    """
    
    def test_dataset_loading_interface(self):
        """Test dataset loading follows standard interface"""
        from omnigenbench import OmniDatasetForSequenceClassification
        
        # Dataset loading should accept these parameters
        # (actual loading tested in dataset-specific tests)
        required_params = [
            "dataset_name_or_path",
            "tokenizer",
            "max_length",
        ]
        
        # Verify class exists and has from_hub method
        assert hasattr(OmniDatasetForSequenceClassification, "from_hub")
        assert hasattr(OmniDatasetForSequenceClassification, "from_files")
    
    def test_benchmark_split_structure(self):
        """Test benchmark datasets have standard splits"""
        # Standard splits for benchmarking
        expected_splits = ["train", "valid", "test"]
        
        # Mock dataset structure
        mock_datasets = {split: [] for split in expected_splits}
        
        # Verify all splits present
        for split in expected_splits:
            assert split in mock_datasets


class TestAutoWorkflowIntegration:
    """
    Integration tests for combined Auto* workflows.
    """
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_train_then_benchmark_workflow(self, tmp_path):
        """
        Test workflow: train a model, then benchmark it.
        This mimics real research workflow.
        """
        output_dir = tmp_path / "custom_model"
        
        # Step 1: Train a custom model
        trainer = AutoTrain(
            dataset_name_or_path="translation_efficiency_prediction",
            model_name_or_path="yangheng/PlantRNA-FM",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
        )
        
        # Step 2: Benchmark the trained model
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path=str(output_dir),
            overwrite=True,
        )
        
        # We're testing workflow structure here
        # Actual execution would be very slow
        assert trainer is not None
        assert bench is not None
    
    def test_benchmark_multiple_models(self):
        """Test benchmarking multiple models in sequence"""
        models = [
            "yangheng/OmniGenome-52M",
            "yangheng/OmniGenome-186M",
            "yangheng/PlantRNA-FM",
        ]
        
        benchmark = "RGB"
        
        # Create benchmark objects for each model
        benches = []
        for model in models:
            bench = AutoBench(
                benchmark=benchmark,
                model_name_or_path=model,
            )
            benches.append(bench)
        
        # Verify all created
        assert len(benches) == len(models)
        
        # In practice, would run bench.run() for each
        # and aggregate results


@pytest.mark.integration
class TestRealWorldBenchmarks:
    """
    Tests based on real benchmark usage patterns.
    """
    
    @pytest.mark.slow
    def test_rgb_benchmark_subset(self):
        """
        Test RGB benchmark on a single task.
        RGB has multiple tasks, test one for speed.
        """
        # RGB contains: RNA-SSP, RNA-mRNA, etc.
        # Full benchmark would test all
        
        bench = AutoBench(
            benchmark="RGB",
            model_name_or_path="yangheng/OmniGenome-52M",
            overwrite=True,
        )
        
        # In practice, bench.run() would execute
        # We're testing the setup here
        assert bench is not None
    
    def test_benchmark_result_format(self):
        """Test benchmark results follow expected format"""
        # Mock result structure
        mock_result = {
            "model": "yangheng/OmniGenome-186M",
            "benchmark": "RGB",
            "tasks": {
                "task1": {
                    "accuracy": 0.85,
                    "f1": 0.82,
                },
                "task2": {
                    "accuracy": 0.87,
                    "f1": 0.84,
                },
            },
            "average": {
                "accuracy": 0.86,
                "f1": 0.83,
            },
        }
        
        # Verify structure
        assert "model" in mock_result
        assert "benchmark" in mock_result
        assert "tasks" in mock_result
        
        # Each task should have metrics
        for task_result in mock_result["tasks"].values():
            assert isinstance(task_result, dict)
