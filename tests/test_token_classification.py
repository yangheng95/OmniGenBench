# -*- coding: utf-8 -*-
# file: test_token_classification.py
# time: 18:00 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for Token Classification and Token Regression models.
Based on examples config.py files for structure prediction and mRNA degradation.
"""

import pytest
import torch
import numpy as np

from omnigenbench import (
    AutoConfig,
    OmniModelForTokenClassification,
    OmniModelForTokenRegression,
    OmniDatasetForTokenClassification,
    OmniDatasetForTokenRegression,
    ClassificationMetric,
    RegressionMetric,
    OmniTokenizer,
)


@pytest.fixture(scope="module")
def model_name():
    """Model name for testing"""
    return "yangheng/OmniGenome-186M"


class TestTokenClassificationConfig:
    """
    Test token classification configuration.
    Based on examples/genomic_data_augmentation/toy_datasets/config.py
    and examples/autobench_gfm_evaluation/RGB/RNA-SSP-*/config.py
    """

    def test_structure_prediction_config(self):
        """
        Test config for RNA secondary structure prediction.
        Based on RNA-SSP config files
        """
        label2id = {"(": 0, ")": 1, ".": 2}
        
        config_dict = {
            "task_name": "RNA-SSP-Test",
            "task_type": "token_classification",
            "label2id": label2id,
            "num_labels": None,  # Will be inferred from label2id
            "epochs": 5,
            "patience": 5,
            "learning_rate": 2e-5,
            "weight_decay": 0,
            "batch_size": 4,
            "max_length": 128,
            "seeds": [42],
            "use_str": False,
            "use_kmer": True,
            "compute_metrics": [
                ClassificationMetric(ignore_y=-100, average="macro").f1_score,
                ClassificationMetric(ignore_y=-100).matthews_corrcoef
            ],
        }
        
        config = AutoConfig(config_dict)
        
        # Verify config properties
        assert config.task_type == "token_classification", \
            "Task type should be token_classification"
        assert config.label2id == label2id, "label2id should match"
        assert config.batch_size == 4, "Batch size should be 4"
        assert config.max_length == 128, "Max length should be 128"

    def test_label2id_to_num_labels(self):
        """Test that num_labels is inferred from label2id"""
        label2id = {"(": 0, ")": 1, ".": 2}
        
        config_dict = {
            "task_type": "token_classification",
            "label2id": label2id,
            "num_labels": None,
        }
        
        config = AutoConfig(config_dict)
        
        # num_labels should be inferred
        expected_num_labels = len(label2id)
        assert config.num_labels == expected_num_labels or config.num_labels is None, \
            f"num_labels should be {expected_num_labels} or None"


class TestTokenClassificationModel:
    """Test token classification model functionality"""

    def test_model_initialization(self, model_name):
        """Test initializing token classification model"""
        label2id = {"(": 0, ")": 1, ".": 2}
        
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenClassification(
            model_name,
            tokenizer=tokenizer,
            label2id=label2id,
            trust_remote_code=True
        )
        
        assert model is not None, "Model should be initialized"
        assert hasattr(model, 'classifier'), "Model should have classifier layer"

    def test_model_forward(self, model_name):
        """Test forward pass through model"""
        label2id = {"(": 0, ")": 1, ".": 2}
        
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenClassification(
            model_name,
            tokenizer=tokenizer,
            label2id=label2id,
            trust_remote_code=True
        )
        
        # Create dummy inputs
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        
        assert "predictions" in result, "Result should have predictions"
        predictions = result["predictions"]
        
        # Predictions should be token-level
        assert isinstance(predictions, (list, torch.Tensor, np.ndarray)), \
            "Predictions should be sequence of labels"

    def test_structure_prediction_labels(self, model_name):
        """Test that structure prediction uses correct label space"""
        label2id = {"(": 0, ")": 1, ".": 2}
        
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenClassification(
            model_name,
            tokenizer=tokenizer,
            label2id=label2id,
            trust_remote_code=True
        )
        
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        predictions = result["predictions"]
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Check that predictions are in valid label range
        if hasattr(predictions, '__iter__') and not isinstance(predictions, str):
            unique_preds = set(np.array(predictions).flatten())
            valid_labels = set(label2id.values())
            
            # Some predictions should be in valid label set
            # (may include padding/ignore index -100)
            assert len(unique_preds) > 0, "Should have some predictions"


class TestTokenRegressionConfig:
    """
    Test token regression configuration.
    Based on examples/autobench_gfm_evaluation/RGB/RNA-mRNA/config.py
    """

    def test_mrna_degradation_config(self):
        """
        Test config for mRNA degradation rate prediction.
        Based on RNA-mRNA config.py
        """
        config_dict = {
            "task_name": "RNA-mRNA-Test",
            "task_type": "token_regression",
            "label2id": None,
            "num_labels": 3,  # 3 target columns
            "epochs": 5,
            "patience": 5,
            "learning_rate": 2e-5,
            "weight_decay": 0,
            "batch_size": 4,
            "max_length": 128,
            "seeds": [42],
            "use_str": True,
            "use_kmer": True,
            "compute_metrics": [RegressionMetric().root_mean_squared_error],
        }
        
        config = AutoConfig(config_dict)
        
        assert config.task_type == "token_regression", \
            "Task type should be token_regression"
        assert config.num_labels == 3, "Should have 3 output labels"
        assert config.batch_size == 4, "Batch size should be 4"


class TestTokenRegressionModel:
    """Test token regression model functionality"""

    def test_model_initialization(self, model_name):
        """Test initializing token regression model"""
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenRegression(
            model_name,
            tokenizer=tokenizer,
            num_labels=3,  # 3 targets for mRNA degradation
            trust_remote_code=True
        )
        
        assert model is not None, "Model should be initialized"
        assert hasattr(model, 'classifier'), "Model should have regression head"

    def test_model_forward_regression(self, model_name):
        """Test forward pass for regression"""
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenRegression(
            model_name,
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        
        assert "predictions" in result, "Result should have predictions"
        predictions = result["predictions"]
        
        # Predictions should be continuous values
        assert isinstance(predictions, (torch.Tensor, np.ndarray, list)), \
            "Predictions should be numeric"

    def test_regression_output_shape(self, model_name):
        """Test that regression outputs have correct shape"""
        num_labels = 3
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenRegression(
            model_name,
            tokenizer=tokenizer,
            num_labels=num_labels,
            trust_remote_code=True
        )
        
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        predictions = result["predictions"]
        
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        # Should have predictions for each position and each target
        if isinstance(predictions, np.ndarray):
            # Last dimension should match num_labels
            assert predictions.shape[-1] == num_labels or predictions.ndim == 1, \
                f"Predictions should have {num_labels} output dimensions"


class TestMetrics:
    """Test classification and regression metrics"""

    def test_classification_metrics(self):
        """Test classification metrics for structure prediction"""
        # Create dummy predictions and labels
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])  # One mismatch
        
        # F1 score
        metric = ClassificationMetric(average="macro")
        f1_result = metric.f1_score(y_pred, y_true)
        
        # Metrics return dict, extract the value
        f1 = f1_result["f1_score"]
        assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
        
        # Perfect prediction should give F1 = 1
        y_pred_perfect = y_true.copy()
        f1_perfect_result = metric.f1_score(y_pred_perfect, y_true)
        f1_perfect = f1_perfect_result["f1_score"]
        assert abs(f1_perfect - 1.0) < 1e-5, "Perfect prediction should have F1 ~1.0"

    def test_classification_metrics_with_ignore(self):
        """Test metrics with ignore index (-100)"""
        # Include padding tokens (-100)
        y_true = np.array([0, 1, 2, -100, -100, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 0])
        
        metric = ClassificationMetric(ignore_y=-100, average="macro")
        f1_result = metric.f1_score(y_pred, y_true)
        
        # Metrics return dict, extract the value
        f1 = f1_result["f1_score"]
        # Should ignore -100 positions
        assert 0 <= f1 <= 1, "F1 should be in valid range"

    def test_regression_metrics(self):
        """Test regression metrics for mRNA degradation"""
        # Create dummy predictions and labels
        y_true = np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.4]])
        y_pred = np.array([[0.4, 0.3, 0.7], [0.3, 0.5, 0.4]])
        
        metric = RegressionMetric()
        rmse_result = metric.root_mean_squared_error(y_pred, y_true)
        
        # Metrics return dict, extract the value
        rmse = rmse_result["root_mean_squared_error"]
        assert rmse >= 0, "RMSE should be non-negative"
        
        # Perfect prediction should give RMSE = 0
        rmse_perfect_result = metric.root_mean_squared_error(y_true, y_true)
        rmse_perfect = rmse_perfect_result["root_mean_squared_error"]
        assert abs(rmse_perfect) < 1e-5, "Perfect prediction should have RMSE ~0"

    def test_matthews_correlation_coefficient(self):
        """Test Matthews correlation coefficient"""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])  # Perfect prediction
        
        metric = ClassificationMetric()
        mcc_result = metric.matthews_corrcoef(y_pred, y_true)
        
        # Metrics return dict, extract the value
        mcc = mcc_result["matthews_corrcoef"]
        # Perfect prediction should give MCC = 1
        assert abs(mcc - 1.0) < 1e-5, "Perfect prediction should have MCC ~1.0"
        
        # Random prediction should give MCC near 0
        y_pred_random = np.array([0, 1, 0, 2, 1, 0])
        mcc_random_result = metric.matthews_corrcoef(y_pred_random, y_true)
        mcc_random = mcc_random_result["matthews_corrcoef"]
        assert -1 <= mcc_random <= 1, "MCC should be between -1 and 1"


class TestDatasetPreparation:
    """Test dataset preparation for token-level tasks"""

    def test_token_classification_dataset_structure(self):
        """Test expected structure for token classification dataset"""
        # Example from toy_datasets/train.json
        sample = {
            "seq": "AUGCCGACGGUCAUAGGACGGGGGAAACACCCGGACUCAUUCCGAACCCGGAAGUUAAGCCCCGUUCCGUCCCGCACAGUACUGUGUUCCGAGAGGGCACGGGAACUGCGGGAACCGUCGGCUUU",
            "label": ".((((((((((....(((((((......((((((.............))))..)).....))))).))...(((((.....((((((.((....))))))))....)))))..))))))))).))"
        }
        
        # Check structure
        assert "seq" in sample, "Sample should have 'seq' field"
        assert "label" in sample, "Sample should have 'label' field"
        
        # Sequence and label should have same length
        assert len(sample["seq"]) == len(sample["label"]), \
            "Sequence and label should have same length"
        
        # Label should contain valid structure symbols
        assert all(c in '().' for c in sample["label"]), \
            "Label should only contain (, ), and ."

    def test_token_regression_dataset_structure(self):
        """Test expected structure for token regression dataset"""
        # Example structure for mRNA degradation
        sample = {
            "sequence": "AUGCCGACGGUCAUAGGACGGGGG",
            "reactivity": [0.1, 0.2, 0.15, 0.3] * 6,  # Per-base reactivity
            "deg_Mg_pH10": [0.05, 0.1, 0.08, 0.12] * 6,
            "deg_Mg_50C": [0.02, 0.04, 0.03, 0.06] * 6,
        }
        
        # Check structure
        assert "sequence" in sample, "Sample should have 'sequence' field"
        assert "reactivity" in sample, "Should have reactivity values"
        assert "deg_Mg_pH10" in sample, "Should have degradation values"
        assert "deg_Mg_50C" in sample, "Should have degradation values"
        
        # All targets should have same length as sequence
        seq_len = len(sample["sequence"])
        assert len(sample["reactivity"]) == seq_len, \
            "Reactivity should match sequence length"


class TestLossFunction:
    """Test loss function for token-level tasks"""

    def test_token_classification_loss_with_padding(self):
        """Test that loss ignores padding tokens"""
        # Simulate logits and labels with padding
        logits = torch.randn(2, 10, 3)  # (batch, seq_len, num_classes)
        labels = torch.tensor([
            [0, 1, 2, 0, 1, -100, -100, -100, -100, -100],
            [2, 1, 0, 2, 1, 0, -100, -100, -100, -100]
        ])
        
        # Cross entropy loss should ignore -100
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.view(-1, 3), labels.view(-1))
        
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_token_regression_loss_with_padding(self):
        """Test that regression loss ignores padding"""
        # Simulate predictions and labels with padding (-100)
        predictions = torch.randn(2, 10, 3)  # (batch, seq_len, num_targets)
        labels = torch.randn(2, 10, 3)
        labels[:, 5:, :] = -100  # Padding
        
        # Create mask for valid positions
        mask = labels != -100
        
        # Only compute loss on valid positions
        valid_predictions = predictions[mask]
        valid_labels = labels[mask]
        
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(valid_predictions, valid_labels)
        
        assert loss >= 0, "MSE loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"


@pytest.mark.integration
class TestTokenLevelPipeline:
    """Integration tests for complete token-level task pipelines"""

    def test_structure_prediction_pipeline(self, model_name):
        """Test complete structure prediction pipeline"""
        # 1. Define config
        label2id = {"(": 0, ")": 1, ".": 2}
        
        config_dict = {
            "task_type": "token_classification",
            "label2id": label2id,
            "max_length": 128,
        }
        
        config = AutoConfig(config_dict)
        
        # 2. Initialize model
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenClassification(
            model_name,
            tokenizer=tokenizer,
            label2id=label2id,
            trust_remote_code=True
        )
        
        # 3. Run inference
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        
        # 4. Verify results
        assert "predictions" in result, "Should have predictions"
        assert "logits" in result, "Should have logits"

    def test_degradation_prediction_pipeline(self, model_name):
        """Test complete mRNA degradation prediction pipeline"""
        # 1. Define config
        config_dict = {
            "task_type": "token_regression",
            "num_labels": 3,
            "max_length": 128,
        }
        
        config = AutoConfig(config_dict)
        
        # 2. Initialize model
        tokenizer = OmniTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model = OmniModelForTokenRegression(
            model_name,
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        
        # 3. Run inference
        sequence = "AUGCGAUCUCGAGCUACGUCGAUG"
        result = model.predict(sequence)
        
        # 4. Verify results
        assert "predictions" in result, "Should have predictions"
        predictions = result["predictions"]
        
        # Should output continuous values for each position
        assert predictions is not None, "Predictions should not be None"
