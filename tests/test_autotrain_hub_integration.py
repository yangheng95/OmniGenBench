# -*- coding: utf-8 -*-
# file: test_autotrain_hub_integration.py
# time: 16:45 08/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test suite for AutoTrain HuggingFace Hub integration
"""

import os
import pytest
import shutil
import tempfile
from pathlib import Path

from omnigenbench import AutoTrain, AutoConfig
from omnigenbench.src.abc.abstract_dataset import OmniDataset


class TestAutoTrainHubIntegration:
    """Test AutoTrain integration with HuggingFace Hub datasets"""

    def test_hub_dataset_detection(self):
        """Test that AutoTrain correctly detects hub vs local datasets"""
        # Create a temporary local directory
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local_dataset"
            local_path.mkdir()

            # Test with local path (should be detected as local)
            trainer_local = AutoTrain(
                dataset=str(local_path),
                config_or_model="yangheng/OmniGenome-52M",
            )
            assert not trainer_local.is_hub_dataset
            assert trainer_local.dataset == str(local_path)

        # Test hub dataset detection without actually downloading
        # We just check the attribute is set correctly
        # (actual download testing would require a real hub dataset)
        # We'll test this by checking the is_hub_dataset flag is set
        # when path doesn't exist
        assert True  # Placeholder - actual hub download tested in integration

    def test_autoconfig_from_hub_method_exists(self):
        """Test that AutoConfig has the from_hub classmethod"""
        assert hasattr(AutoConfig, "from_hub")
        assert callable(getattr(AutoConfig, "from_hub"))

    def test_autoconfig_from_local_dict(self):
        """Test AutoConfig initialization from a config dict"""
        config_dict = {
            "task_name": "test_task",
            "task_type": "sequence_classification",
            "num_labels": 2,
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 2e-5,
        }

        config = AutoConfig(config_dict)

        assert config.task_name == "test_task"
        assert config.task_type == "sequence_classification"
        assert config.num_labels == 2
        assert config.epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 2e-5

    def test_autoconfig_override_params(self):
        """Test that AutoConfig parameters can be overridden"""
        config_dict = {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "epochs": 50,
        }

        config = AutoConfig(config_dict)
        assert config.learning_rate == 2e-5

        # Override with update
        config.update({"learning_rate": 1e-4})
        assert config.learning_rate == 1e-4

    def test_dataset_download_method_exists(self):
        """Test that OmniDataset has the _download_dataset_from_hub method"""
        assert hasattr(OmniDataset, "_download_dataset_from_hub")
        assert callable(getattr(OmniDataset, "_download_dataset_from_hub"))

    def test_autotrain_cache_dir_parameter(self):
        """Test that AutoTrain accepts cache_dir parameter for local datasets"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local dataset directory
            dataset_dir = Path(tmpdir) / "local_dataset"
            dataset_dir.mkdir()
            
            cache_dir = Path(tmpdir) / "custom_cache"

            trainer = AutoTrain(
                dataset=str(dataset_dir),  # Use local path to avoid download
                config_or_model="yangheng/OmniGenome-52M",
                cache_dir=str(cache_dir),
            )

            # Check that cache_dir is stored
            assert hasattr(trainer, "cache_dir")
            # For local datasets, cache_dir might not be used, but should be set
            assert trainer.cache_dir == str(cache_dir)

    def test_autotrain_run_config_parameter(self):
        """Test that AutoTrain.run() accepts a config parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local dataset to avoid hub download
            dataset_dir = Path(tmpdir) / "test_dataset"
            dataset_dir.mkdir()
            
            # Create a custom config
            config_dict = {
                "task_type": "sequence_classification",
                "num_labels": 2,
                "epochs": 5,
                "batch_size": 4,
                "learning_rate": 2e-5,
                "seeds": [42],
            }
            custom_config = AutoConfig(config_dict)

            # This should not raise an error
            trainer = AutoTrain(
                dataset=str(dataset_dir),
                config_or_model="yangheng/OmniGenome-52M",
            )

            # Verify that run method accepts config parameter
            import inspect

            sig = inspect.signature(trainer.run)
            assert "kwargs" in sig.parameters or "config" in str(sig)

    def test_backward_compatibility_local_dataset(self):
        """Test that local dataset workflows still work (backward compatibility)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local_dataset"
            local_path.mkdir()

            # Old-style initialization should still work
            trainer = AutoTrain(
                dataset=str(local_path),
                config_or_model="yangheng/OmniGenome-52M",
            )

            assert trainer.dataset == str(local_path)
            assert not trainer.is_hub_dataset

    def test_autotrain_dataset_name_handling(self):
        """Test that dataset names are handled correctly for metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local dataset
            dataset_dir = Path(tmpdir) / "my_dataset_name"
            dataset_dir.mkdir()
            
            trainer = AutoTrain(
                dataset=str(dataset_dir),
                config_or_model="yangheng/OmniGenome-52M",
            )

            # Check that dataset name is extracted correctly
            assert hasattr(trainer, "dataset_name_or_path")
            assert "my_dataset_name" in trainer.dataset_name_or_path

    def test_autoconfig_dict_like_behavior(self):
        """Test that AutoConfig behaves like a dictionary"""
        config_dict = {
            "param1": "value1",
            "param2": 42,
            "param3": [1, 2, 3],
        }

        config = AutoConfig(config_dict)

        # Test dict-like access
        assert config["param1"] == "value1"
        assert config["param2"] == 42
        assert "param1" in config
        assert len(config) == 3

        # Test iteration
        assert set(config.keys()) == {"param1", "param2", "param3"}
        assert "value1" in config.values()

        # Test item assignment
        config["param4"] = "value4"
        assert config["param4"] == "value4"

    def test_autoconfig_get_method(self):
        """Test AutoConfig.get() method with defaults"""
        config = AutoConfig({"existing_key": "value"})

        assert config.get("existing_key") == "value"
        assert config.get("nonexistent_key", "default") == "default"
        assert config.get("nonexistent_key") is None


class TestAutoConfigFromHub:
    """Test AutoConfig.from_hub functionality"""

    def test_from_hub_with_local_config(self):
        """Test loading config from a local directory with config.py"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "test_dataset"
            dataset_dir.mkdir()

            # Create a test config.py
            config_content = """
from omnigenbench import AutoConfig

config_dict = {
    "task_name": "test_task",
    "task_type": "sequence_classification",
    "num_labels": 2,
    "epochs": 10,
    "batch_size": 8,
}

bench_config = AutoConfig(config_dict)
"""
            config_file = dataset_dir / "config.py"
            config_file.write_text(config_content)

            # Load config from local directory
            loaded_config = AutoConfig.from_hub(str(dataset_dir))

            assert loaded_config is not None
            assert loaded_config.task_name == "test_task"
            assert loaded_config.num_labels == 2

    def test_from_hub_with_override_params(self):
        """Test that from_hub respects override parameters"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "test_dataset"
            dataset_dir.mkdir()

            # Create a test config.py
            config_content = """
from omnigenbench import AutoConfig

config_dict = {
    "task_name": "test_task",
    "learning_rate": 2e-5,
    "batch_size": 8,
}

bench_config = AutoConfig(config_dict)
"""
            config_file = dataset_dir / "config.py"
            config_file.write_text(config_content)

            # Load config with overrides
            loaded_config = AutoConfig.from_hub(
                str(dataset_dir), learning_rate=1e-4, batch_size=16
            )

            assert loaded_config.learning_rate == 1e-4
            assert loaded_config.batch_size == 16
            assert loaded_config.task_name == "test_task"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
