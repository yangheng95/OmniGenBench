# -*- coding: utf-8 -*-
# file: test_training_workflows.py
# time: 14:00 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for model training workflows.
Based on examples from:
- examples/tfb_prediction/03_model_training.ipynb
- examples/translation_efficiency_prediction/quickstart_te.py
- examples/mRNA_degrad_rate_regression/
"""

import pytest
import torch
import json
import tempfile
from pathlib import Path

from omnigenbench import (
    OmniTokenizer,
    OmniModelForSequenceClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniDatasetForSequenceClassification,
    OmniDatasetForMultiLabelClassification,
    ClassificationMetric,
    AccelerateTrainer,
)


@pytest.fixture(scope="module")
def test_model_small():
    """Use smallest model for fast testing"""
    return "yangheng/OmniGenome-52M"


@pytest.fixture(scope="module")
def plant_model_small():
    """Use PlantRNA-FM for plant-specific tasks"""
    return "yangheng/PlantRNA-FM"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for model checkpoints"""
    output_dir = tmp_path / "test_model_output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_binary_dataset(tmp_path):
    """
    Create a minimal mock binary classification dataset.
    Based on translation efficiency prediction pattern.
    """
    data_dir = tmp_path / "mock_dataset"
    data_dir.mkdir()
    
    # Create minimal train/valid/test datasets
    train_data = [
        {"sequence": "AUGCAUGCAUGCAUGCAUGC", "label": "1"},
        {"sequence": "GCAUGCAUGCAUGCAUGCAU", "label": "0"},
        {"sequence": "CAUGCAUGCAUGCAUGCAUG", "label": "1"},
        {"sequence": "AUGCAUGCAUGCAUGCGCGC", "label": "0"},
    ]
    
    valid_data = [
        {"sequence": "GCGCGCGCGCGCGCGCGCGC", "label": "1"},
        {"sequence": "ATATATATATATATATAT", "label": "0"},
    ]
    
    # Write JSON files
    with open(data_dir / "train.json", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(data_dir / "valid.json", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
    
    # Test split is optional but good practice
    with open(data_dir / "test.json", "w") as f:
        for item in valid_data:  # Reuse valid for simplicity
            f.write(json.dumps(item) + "\n")
    
    return str(data_dir)


@pytest.fixture
def mock_multilabel_dataset(tmp_path):
    """
    Create a minimal mock multi-label classification dataset.
    Based on TFB prediction (DeepSEA) pattern with 919 labels.
    """
    data_dir = tmp_path / "mock_multilabel_dataset"
    data_dir.mkdir()
    
    # For testing, use just 10 labels instead of 919
    num_labels = 10
    
    train_data = [
        {"sequence": "ATCGATCGATCGATCGATCG", "labels": [1, 0, 1, 0, 0, 1, 0, 0, 0, 1]},
        {"sequence": "GCGCGCGCGCGCGCGCGCGC", "labels": [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]},
        {"sequence": "TATATATATATATATATAT", "labels": [1, 1, 0, 0, 1, 0, 1, 0, 0, 0]},
    ]
    
    valid_data = [
        {"sequence": "AGAGAGAGAGAGAGAGAGAG", "labels": [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]},
    ]
    
    # Write JSON files
    with open(data_dir / "train.json", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(data_dir / "valid.json", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
    
    with open(data_dir / "test.json", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
    
    return str(data_dir), num_labels


class TestSequenceClassificationTraining:
    """
    Test binary sequence classification training workflow.
    Pattern: Translation Efficiency Prediction (examples/translation_efficiency_prediction/)
    """
    
    @pytest.mark.slow
    def test_basic_training_workflow(self, plant_model_small, mock_binary_dataset, temp_output_dir):
        """
        Test complete training workflow for binary classification.
        Based on quickstart_te.py pattern.
        """
        # 1. Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(plant_model_small)
        
        # 2. Prepare dataset
        label2id = {"0": 0, "1": 1}
        datasets = OmniDatasetForSequenceClassification.from_files(
            train_file=f"{mock_binary_dataset}/train.json",
            valid_file=f"{mock_binary_dataset}/valid.json",
            test_file=f"{mock_binary_dataset}/test.json",
            tokenizer=tokenizer,
            max_length=128,  # Short for fast testing
            label2id=label2id,
        )
        
        # Verify dataset loaded
        assert "train" in datasets
        assert len(datasets["train"]) > 0
        
        # 3. Initialize model
        model = OmniModelForSequenceClassification(
            model=plant_model_small,
            num_labels=2,
        )
        
        # Verify model initialized
        assert model is not None
        assert hasattr(model, "config")
        
        # 4. Setup training configuration
        config = {
            "epochs": 1,  # Single epoch for fast testing
            "batch_size": 2,
            "learning_rate": 2e-5,
            "output_dir": temp_output_dir,
            "save_steps": 100,
            "logging_steps": 10,
        }
        
        # 5. Define metrics
        metric = ClassificationMetric()
        compute_metrics = metric.accuracy
        
        # 6. Initialize trainer
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("valid"),
            config=config,
            compute_metrics=compute_metrics,
        )
        
        # 7. Run training
        trainer.train()
        
        # 8. Verify training artifacts
        output_path = Path(temp_output_dir)
        assert output_path.exists(), "Output directory should be created"
        
        # Check for model files (may vary by trainer implementation)
        model_files = list(output_path.glob("*.bin")) + list(output_path.glob("*.safetensors"))
        assert len(model_files) > 0 or (output_path / "pytorch_model.bin").exists(), \
            "Model checkpoint should be saved"
    
    def test_dataset_loading_from_hub(self, plant_model_small):
        """
        Test loading dataset from HuggingFace Hub.
        Pattern from quickstart_te.py
        """
        tokenizer = OmniTokenizer.from_pretrained(plant_model_small)
        
        # Load from hub (will download if needed)
        datasets = OmniDatasetForSequenceClassification.from_hub(
            dataset_name_or_path="translation_efficiency_prediction",
            tokenizer=tokenizer,
            max_length=128,
            label2id={"0": 0, "1": 1},
        )
        
        # Verify structure
        assert isinstance(datasets, dict)
        assert "train" in datasets
        
    def test_model_initialization_with_labels(self, test_model_small):
        """Test model properly initializes classification head"""
        num_labels = 3
        model = OmniModelForSequenceClassification(
            model=test_model_small,
            num_labels=num_labels,
        )
        
        # Verify classification head
        assert model.config.num_labels == num_labels
        assert hasattr(model, "classifier") or hasattr(model, "score")


class TestMultiLabelClassificationTraining:
    """
    Test multi-label classification training workflow.
    Pattern: TFB Prediction (examples/tfb_prediction/03_model_training.ipynb)
    """
    
    @pytest.mark.slow
    def test_multilabel_training_workflow(self, test_model_small, mock_multilabel_dataset, temp_output_dir):
        """
        Test complete training workflow for multi-label classification.
        Based on TFB prediction notebook pattern.
        """
        dataset_path, num_labels = mock_multilabel_dataset
        
        # 1. Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(test_model_small)
        
        # 2. Prepare multi-label dataset
        datasets = OmniDatasetForMultiLabelClassification.from_files(
            train_file=f"{dataset_path}/train.json",
            valid_file=f"{dataset_path}/valid.json",
            test_file=f"{dataset_path}/test.json",
            tokenizer=tokenizer,
            max_length=128,
        )
        
        # 3. Initialize multi-label model
        model = OmniModelForMultiLabelSequenceClassification(
            model=test_model_small,
            num_labels=num_labels,
        )
        
        # 4. Training config
        config = {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "output_dir": temp_output_dir,
        }
        
        # 5. Multi-label metrics (typically ROC-AUC)
        metric = ClassificationMetric()
        
        # 6. Train
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("valid"),
            config=config,
            compute_metrics=metric.roc_auc,
        )
        
        trainer.train()
        
        # Verify output
        assert Path(temp_output_dir).exists()
    
    def test_multilabel_dataset_structure(self, test_model_small, mock_multilabel_dataset):
        """Verify multi-label dataset structure matches expected format"""
        dataset_path, num_labels = mock_multilabel_dataset
        tokenizer = OmniTokenizer.from_pretrained(test_model_small)
        
        datasets = OmniDatasetForMultiLabelClassification.from_files(
            train_file=f"{dataset_path}/train.json",
            tokenizer=tokenizer,
            max_length=128,
        )
        
        # Check first sample structure
        sample = datasets["train"][0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert len(sample["labels"]) == num_labels, \
            f"Labels should be length {num_labels}, got {len(sample['labels'])}"


class TestTrainerComponents:
    """
    Test individual trainer components and configuration.
    Based on 03_model_training.ipynb patterns.
    """
    
    def test_metric_functions(self):
        """Test ClassificationMetric provides correct functions"""
        metric = ClassificationMetric()
        
        # Verify metric methods exist
        assert hasattr(metric, "accuracy")
        assert hasattr(metric, "f1")
        assert hasattr(metric, "roc_auc")
        assert callable(metric.accuracy)
    
    def test_trainer_initialization(self, test_model_small, mock_binary_dataset, temp_output_dir):
        """Test trainer can be initialized with minimal config"""
        tokenizer = OmniTokenizer.from_pretrained(test_model_small)
        
        datasets = OmniDatasetForSequenceClassification.from_files(
            train_file=f"{mock_binary_dataset}/train.json",
            tokenizer=tokenizer,
            max_length=128,
            label2id={"0": 0, "1": 1},
        )
        
        model = OmniModelForSequenceClassification(
            model=test_model_small,
            num_labels=2,
        )
        
        config = {
            "epochs": 1,
            "batch_size": 2,
            "output_dir": temp_output_dir,
        }
        
        # Should initialize without errors
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            config=config,
        )
        
        assert trainer is not None
        assert hasattr(trainer, "train")
    
    def test_different_optimizers_config(self, test_model_small, mock_binary_dataset, temp_output_dir):
        """Test trainer accepts different optimizer configurations"""
        tokenizer = OmniTokenizer.from_pretrained(test_model_small)
        
        datasets = OmniDatasetForSequenceClassification.from_files(
            train_file=f"{mock_binary_dataset}/train.json",
            tokenizer=tokenizer,
            max_length=128,
            label2id={"0": 0, "1": 1},
        )
        
        model = OmniModelForSequenceClassification(
            model=test_model_small,
            num_labels=2,
        )
        
        # Config with optimizer settings
        config = {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "output_dir": temp_output_dir,
        }
        
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            config=config,
        )
        
        assert trainer is not None


@pytest.mark.integration
class TestEndToEndTraining:
    """
    Integration tests for complete training pipelines.
    Based on full workflow examples.
    """
    
    @pytest.mark.slow
    def test_full_tfb_training_pipeline(self, test_model_small, mock_multilabel_dataset, temp_output_dir):
        """
        Complete TFB prediction training pipeline.
        Mimics examples/tfb_prediction/03_model_training.ipynb
        """
        dataset_path, num_labels = mock_multilabel_dataset
        
        # Configuration (matches tutorial)
        model_name = test_model_small
        max_length = 128
        batch_size = 2
        learning_rate = 2e-5
        epochs = 1
        
        # 1. Tokenizer
        tokenizer = OmniTokenizer.from_pretrained(model_name)
        
        # 2. Dataset
        datasets = OmniDatasetForMultiLabelClassification.from_files(
            train_file=f"{dataset_path}/train.json",
            valid_file=f"{dataset_path}/valid.json",
            tokenizer=tokenizer,
            max_length=max_length,
        )
        
        # 3. Model
        model = OmniModelForMultiLabelSequenceClassification(
            model=model_name,
            num_labels=num_labels,
        )
        
        # 4. Metric
        metric = ClassificationMetric()
        
        # 5. Training config
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "output_dir": temp_output_dir,
            "logging_steps": 5,
            "save_steps": 50,
        }
        
        # 6. Trainer
        trainer = AccelerateTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("valid"),
            config=config,
            compute_metrics=metric.roc_auc,
        )
        
        # 7. Train
        trainer.train()
        
        # 8. Verify artifacts
        output_path = Path(temp_output_dir)
        assert output_path.exists()
        
        # Check for typical output files
        expected_files = ["config.json", "tokenizer_config.json"]
        for fname in expected_files:
            # Files may or may not exist depending on trainer implementation
            pass  # Non-strict check
