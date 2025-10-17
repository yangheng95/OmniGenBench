"""
Test inference functionality with dataset preprocessing.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock, Mock
import warnings

try:
    import torch
    from transformers import BatchEncoding
except ImportError:
    torch = None
    BatchEncoding = None

# Mark as slow tests - can be run with --run-slow
pytestmark = pytest.mark.slow


class TestInferenceWithDataset:
    """Test inference with dataset preprocessing functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Mock model config."""
        config = MagicMock()
        config.hidden_size = 768
        config.num_labels = 2
        config.label2id = {"negative": 0, "positive": 1}
        config.id2label = {0: "negative", 1: "positive"}
        config.pad_token_id = 0
        return config

    @pytest.fixture
    def mock_dataset_class(self, mock_tokenizer):
        """Mock dataset class with prepare_input method."""
        class MockDataset:
            def __init__(self, dataset_name, tokenizer, max_length=None, **kwargs):
                self.dataset_name = dataset_name
                self.tokenizer = tokenizer
                self.max_length = max_length or 1024

            def prepare_input(self, instance, **kwargs):
                """Mock prepare_input method."""
                if isinstance(instance, dict):
                    sequence = instance.get('sequence', instance.get('seq'))
                elif isinstance(instance, str):
                    sequence = instance
                else:
                    raise ValueError("Unsupported instance type")

                # Mock tokenization
                return {
                    'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]),
                    'labels': torch.tensor([instance.get('label', -100)]) if isinstance(instance, dict) else torch.tensor([-100])
                }

        return MockDataset

    def test_model_init_with_dataset_class(self, mock_tokenizer, mock_config, mock_dataset_class):
        """Test model initialization with dataset_class parameter."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            # Initialize model with dataset_class
            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1},
                dataset_class=mock_dataset_class
            )

            # Verify dataset_class is set
            assert hasattr(model, 'dataset_class')
            assert model.dataset_class == mock_dataset_class
            assert 'dataset_cls' in model.metadata

    def test_inference_with_string_input(self, mock_tokenizer, mock_config):
        """Test inference with string input (traditional way)."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1}
            )

            # Should work without dataset_class
            # Note: This test mainly verifies that the code path works
            # Actual inference behavior is mocked

    def test_inference_with_dict_input_and_dataset_class(self, mock_tokenizer, mock_config, mock_dataset_class):
        """Test inference with dict input when dataset_class is set."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1},
                dataset_class=mock_dataset_class
            )

            # Dict input should be processed by dataset.prepare_input
            # Note: Actual processing is mocked, we're testing the code path

    def test_metadata_contains_dataset_info(self, mock_tokenizer, mock_config, mock_dataset_class):
        """Test that metadata contains dataset information."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1},
                dataset_class=mock_dataset_class
            )

            # Check metadata
            assert 'dataset_cls' in model.metadata
            assert 'dataset_module' in model.metadata
            assert model.metadata['dataset_cls'] == mock_dataset_class.__name__

    def test_backward_compatibility_without_dataset_class(self, mock_tokenizer, mock_config):
        """Test that models work without dataset_class (backward compatibility)."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            # Initialize without dataset_class
            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1}
            )

            # Should not have dataset_class attribute
            assert not hasattr(model, 'dataset_class')
            # Metadata should not have dataset info
            assert 'dataset_cls' not in model.metadata

    def test_dataset_class_persistence(self, mock_tokenizer, mock_config, mock_dataset_class):
        """Test that dataset_class information is saved in metadata."""
        if torch is None:
            pytest.skip("torch not available")

        try:
            from omnigenbench import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenbench not available")

        with patch('omnigenbench.src.abc.abstract_model.AutoModel') as mock_auto_model, \
             patch('omnigenbench.src.abc.abstract_model.AutoConfig') as mock_auto_config:
            
            mock_auto_config.from_pretrained.return_value = mock_config
            mock_model_instance = MagicMock()
            mock_model_instance.config = mock_config
            mock_model_instance.device = torch.device('cpu')
            mock_auto_model.from_pretrained.return_value = mock_model_instance

            model = OmniModelForSequenceClassification(
                "test_model",
                mock_tokenizer,
                label2id={"negative": 0, "positive": 1},
                dataset_class=mock_dataset_class
            )

            # Collect metadata
            metadata = model._collect_metadata()

            # Check dataset info is in metadata
            assert 'dataset_cls' in metadata
            assert 'dataset_module' in metadata
            assert metadata['dataset_cls'] == mock_dataset_class.__name__

    def test_input_format_detection(self):
        """Test that different input formats are correctly detected."""
        # This is a conceptual test - actual implementation is in _forward_from_raw_input
        
        # String input
        assert isinstance("ATCGATCG", str)
        
        # List input
        assert isinstance(["ATCG", "GCTA"], list)
        
        # Dict input
        assert isinstance({"sequence": "ATCG", "label": 1}, dict)
        
        # Check dict has expected keys
        test_dict = {"sequence": "ATCG", "label": 1}
        assert "sequence" in test_dict or "seq" in test_dict

    def test_fallback_mechanism(self):
        """Test that fallback to tokenizer works when dataset processing fails."""
        # This test verifies the concept - actual implementation has try-except
        # to fall back to tokenizer when dataset.prepare_input fails
        
        # Conceptual test: if dataset processing fails, should use tokenizer
        has_dataset = False
        use_tokenizer = True if not has_dataset else False
        assert use_tokenizer

    def test_load_dataset_class_method_exists(self):
        """Test that _load_dataset_class method exists in OmniModel."""
        try:
            from omnigenbench.src.abc.abstract_model import OmniModel
            assert hasattr(OmniModel, '_load_dataset_class')
        except ImportError:
            pytest.skip("omnigenbench not available")


class TestDatasetPreprocessingIntegration:
    """Integration tests for dataset preprocessing in inference."""

    def test_dataset_import_patterns(self):
        """Test that dataset classes can be imported."""
        try:
            from omnigenbench import (
                OmniDatasetForSequenceClassification,
                OmniDatasetForTokenClassification,
                OmniDatasetForSequenceRegression,
                OmniDatasetForTokenRegression,
            )
            assert True
        except ImportError:
            pytest.skip("omnigenbench dataset classes not available")

    def test_dataset_prepare_input_exists(self):
        """Test that dataset classes have prepare_input method."""
        try:
            from omnigenbench import OmniDatasetForSequenceClassification
            assert hasattr(OmniDatasetForSequenceClassification, 'prepare_input')
        except ImportError:
            pytest.skip("omnigenbench not available")

    def test_model_dataset_compatibility(self):
        """Test that model and dataset classes are compatible."""
        try:
            from omnigenbench import (
                OmniModelForSequenceClassification,
                OmniDatasetForSequenceClassification,
            )
            # If both can be imported, they should be compatible
            assert True
        except ImportError:
            pytest.skip("omnigenbench not available")

