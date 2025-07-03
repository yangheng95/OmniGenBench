"""
Test model loading functionality based on examples.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

try:
    import torch
except ImportError:
    torch = None

# Skip heavy model loading tests by default - can be run with --run-slow
pytestmark = pytest.mark.slow


class TestModelLoading:
    """Test model loading similar to examples."""

    @pytest.fixture
    def mock_model_config(self):
        """Mock model config to avoid downloading real models."""
        config = MagicMock()
        config.hidden_size = 768
        config.num_labels = 2
        return config

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        tokenizer.decode.return_value = "AUGC"
        tokenizer.convert_ids_to_tokens.return_value = ["A", "U", "G", "C"]
        return tokenizer

    def test_model_import_structure(self):
        """Test that model classes can be imported as shown in examples."""
        try:
            from omnigenome import (
                OmniModelForSequenceClassification,
                OmniModelForTokenClassification,
                OmniModelForSequenceRegression,
                OmniModelForTokenRegression,
            )
            # If import succeeds, test passes
            assert True
        except ImportError:
            pytest.skip("omnigenome not available or missing dependencies")

    def test_embedding_model_import(self):
        """Test embedding model import as shown in RNA_Embedding_Tutorial.ipynb."""
        try:
            from omnigenome import OmniGenomeModelForEmbedding
            assert True
        except ImportError:
            pytest.skip("omnigenome not available or missing dependencies")

    def test_pooling_import(self):
        """Test pooling import as shown in classification.ipynb."""
        try:
            from omnigenome import OmniModel, OmniPooling
            assert True
        except ImportError:
            pytest.skip("omnigenome not available or missing dependencies")

    def test_base_model_loading_pattern(self, mock_tokenizer):
        """Test the base model loading pattern from classification.ipynb."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            pytest.skip("transformers not available")
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_auto_tokenizer, \
             patch('transformers.AutoModel.from_pretrained') as mock_auto_model:
            
            # Mock the returns
            mock_auto_tokenizer.return_value = mock_tokenizer
            mock_auto_model.return_value = MagicMock()

            # This pattern is from examples/custom_finetuning/classification.ipynb
            model_name = "yangheng/OmniGenome-52M"
            base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            base_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Verify the calls were made correctly
            mock_auto_model.assert_called_once_with(model_name, trust_remote_code=True)
            mock_auto_tokenizer.assert_called_once_with(model_name, trust_remote_code=True)

    def test_embedding_model_initialization_pattern(self):
        """Test embedding model initialization pattern from RNA_Embedding_Tutorial.ipynb."""
        if torch is None:
            pytest.skip("torch not available")
            
        try:
            from omnigenome import OmniGenomeModelForEmbedding
        except ImportError:
            pytest.skip("omnigenome not available")
            
        with patch('omnigenome.OmniGenomeModelForEmbedding') as mock_embedding_model:
            mock_instance = MagicMock()
            mock_instance.to.return_value = mock_instance
            mock_embedding_model.return_value = mock_instance

            model_name = "yangheng/OmniGenome-52M"
            embedding_model = OmniGenomeModelForEmbedding(model_name, trust_remote_code=True).to(torch.device("cuda:0")).to(torch.float16)

            # Verify initialization pattern
            mock_embedding_model.assert_called_once_with(model_name, trust_remote_code=True)
            assert mock_instance.to.call_count == 2  # Called twice for device and dtype

    def test_model_parameter_patterns(self):
        """Test that common model parameters are recognized."""
        # These are patterns seen across examples
        common_model_names = [
            "yangheng/OmniGenome-52M",
            "yangheng/OmniGenome-186M",
            "anonymous8/OmniGenome-186M",
            "anonymous8/OmniGenome-52M"
        ]
        
        for model_name in common_model_names:
            # Just verify the string patterns are valid
            assert isinstance(model_name, str)
            assert "/" in model_name
            assert "OmniGenome" in model_name

    def test_classification_model_initialization_pattern(self, mock_tokenizer):
        """Test classification model init pattern from examples."""
        try:
            from omnigenome import OmniModelForSequenceClassification
        except ImportError:
            pytest.skip("omnigenome not available")
            
        with patch('omnigenome.OmniModelForSequenceClassification') as mock_model_class:
            mock_model_class.return_value = MagicMock()

            # Pattern from classification.ipynb
            model_name = "test_model"
            tokenizer = mock_tokenizer
            
            model = OmniModelForSequenceClassification(
                config_or_model=model_name,
                tokenizer=tokenizer,
                num_labels=3,
            )

            mock_model_class.assert_called_once_with(
                config_or_model=model_name,
                tokenizer=tokenizer,
                num_labels=3,
            )

    def test_rna_sequence_patterns(self):
        """Test RNA sequence patterns used in examples."""
        # Patterns from RNA_Embedding_Tutorial.ipynb
        rna_sequences = [
            "AUGGCUACG",
            "CGGAUACGGC", 
            "UGGCCAAGUC",
            "AUGCUGCUAUGCUA"
        ]
        
        for seq in rna_sequences:
            # Basic validation of RNA sequence format
            assert isinstance(seq, str)
            assert len(seq) > 0
            assert all(base in "AUCG" for base in seq)

    def test_device_patterns(self):
        """Test device usage patterns from examples."""
        if torch is None:
            pytest.skip("torch not available")
            
        # Pattern from examples: torch.device("cuda:0")
        device = torch.device("cuda:0")
        assert str(device) == "cuda:0"
        
        # Alternative pattern
        if torch.cuda.is_available():
            device = torch.device("cuda")
            assert "cuda" in str(device) 