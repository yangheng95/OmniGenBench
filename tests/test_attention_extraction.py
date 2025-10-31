# -*- coding: utf-8 -*-
# file: test_attention_extraction.py
# time: 17:35 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for Attention Score Extraction functionality.
Based on examples/attention_score_extraction/attention_extraction_example.py
"""

import pytest
import torch

from omnigenbench import (
    OmniModelForEmbedding,
    OmniModelForSequenceClassification,
    OmniModelForSequenceRegression,
)


@pytest.fixture(scope="module")
def model_name():
    """Model name for testing"""
    return "yangheng/OmniGenome-186M"


@pytest.fixture(scope="module")
def test_sequences():
    """Example genomic sequences from attention_extraction_example.py"""
    return [
        "ATCGATCGATCGTAGCTAGCTAGCT",
        "GGCCTTAACCGGTTAACCGGTTAA",
        "TTTTAAAACCCCGGGGTTTTAAAA"
    ]


class TestAttentionExtractionEmbeddingModel:
    """
    Test attention extraction with OmniModelForEmbedding.
    Based on main() function from attention_extraction_example.py
    """

    @pytest.fixture(scope="class")
    def embedding_model(self, model_name):
        """Load embedding model for attention extraction"""
        model = OmniModelForEmbedding(model=model_name, trust_remote_code=True)
        return model

    def test_single_sequence_attention_extraction(self, embedding_model, test_sequences):
        """
        Test extracting attention scores from a single sequence.
        Based on section 1 of attention_extraction_example.py
        """
        sequence = test_sequences[0]
        
        # Extract attention scores
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=None,  # Extract all layers
            head_indices=None,   # Extract all heads
            return_on_cpu=True
        )
        
        # Verify result structure
        assert "attentions" in attention_result, "Should have 'attentions' key"
        assert "tokens" in attention_result, "Should have 'tokens' key"
        assert "attention_mask" in attention_result, "Should have 'attention_mask' key"
        
        # Check attention tensor shape
        attentions = attention_result['attentions']
        assert isinstance(attentions, torch.Tensor), "Attentions should be a tensor"
        assert attentions.ndim == 4, "Attentions should be 4D: (layers, heads, seq_len, seq_len)"
        
        # Check tokens
        tokens = attention_result['tokens']
        assert isinstance(tokens, list), "Tokens should be a list"
        assert len(tokens) > 0, "Should have at least one token"
        
        # Check attention mask
        attention_mask = attention_result['attention_mask']
        assert isinstance(attention_mask, torch.Tensor), "Attention mask should be a tensor"

    def test_layer_selection(self, embedding_model, test_sequences):
        """Test extracting specific layers only"""
        sequence = test_sequences[0]
        
        # Extract only first and last layer
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=[0, -1],  # First and last layer
            head_indices=None,
            return_on_cpu=True
        )
        
        attentions = attention_result['attentions']
        # Should have 2 layers (first and last)
        assert attentions.shape[0] == 2, "Should extract exactly 2 layers"

    def test_head_selection(self, embedding_model, test_sequences):
        """Test extracting specific attention heads only"""
        sequence = test_sequences[0]
        
        # Extract only first 3 heads
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=None,
            head_indices=[0, 1, 2],  # First 3 heads
            return_on_cpu=True
        )
        
        attentions = attention_result['attentions']
        # Should have 3 heads
        assert attentions.shape[1] == 3, "Should extract exactly 3 heads"

    def test_attention_statistics(self, embedding_model, test_sequences):
        """
        Test attention statistics calculation.
        Based on section 1 of attention_extraction_example.py
        """
        sequence = test_sequences[0]
        
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            return_on_cpu=True
        )
        
        # Get attention statistics
        stats = embedding_model.get_attention_statistics(
            attention_result['attentions'],
            attention_result['attention_mask']
        )
        
        # Verify statistics structure
        assert "attention_matrix" in stats, "Should have attention_matrix"
        assert "attention_entropy" in stats, "Should have attention_entropy"
        assert "attention_concentration" in stats, "Should have attention_concentration"
        
        # Check shapes and values
        assert isinstance(stats['attention_matrix'], torch.Tensor), "Matrix should be tensor"
        assert isinstance(stats['attention_entropy'], torch.Tensor), "Entropy should be tensor"
        assert isinstance(stats['attention_concentration'], torch.Tensor), "Concentration should be tensor"
        
        # Entropy should be positive
        assert (stats['attention_entropy'] >= 0).all(), "Entropy should be non-negative"
        
        # Concentration should be between 0 and 1
        assert (stats['attention_concentration'] >= 0).all(), "Concentration should be >= 0"
        assert (stats['attention_concentration'] <= 1).all(), "Concentration should be <= 1"


class TestAttentionExtractionBatch:
    """
    Test batch attention extraction.
    Based on section 2 of attention_extraction_example.py
    """

    @pytest.fixture(scope="class")
    def embedding_model(self, model_name):
        """Load embedding model for batch extraction"""
        model = OmniModelForEmbedding(model=model_name, trust_remote_code=True)
        return model

    def test_batch_attention_extraction(self, embedding_model, test_sequences):
        """
        Test batch extraction of attention scores.
        Based on section 2 of attention_extraction_example.py
        """
        # Extract attention for all sequences
        batch_results = embedding_model.batch_extract_attention_scores(
            sequences=test_sequences,
            batch_size=2,
            max_length=128,
            layer_indices=[0, -1],  # First and last layer only
            head_indices=[0, 1, 2], # First 3 heads only
            return_on_cpu=True
        )
        
        # Verify results
        assert len(batch_results) == len(test_sequences), \
            f"Should have {len(test_sequences)} results"
        
        for i, result in enumerate(batch_results):
            assert "attentions" in result, f"Result {i} should have attentions"
            assert "tokens" in result, f"Result {i} should have tokens"
            assert "attention_mask" in result, f"Result {i} should have attention_mask"
            
            # Check dimensions
            attentions = result['attentions']
            assert attentions.shape[0] == 2, "Should have 2 layers"
            assert attentions.shape[1] == 3, "Should have 3 heads"

    def test_batch_with_different_lengths(self, embedding_model):
        """Test batch processing with sequences of different lengths"""
        sequences = [
            "ATCG" * 5,      # 20 bases
            "GCTA" * 10,     # 40 bases
            "TTAA" * 15,     # 60 bases
        ]
        
        batch_results = embedding_model.batch_extract_attention_scores(
            sequences=sequences,
            batch_size=3,
            max_length=128,
            return_on_cpu=True
        )
        
        assert len(batch_results) == len(sequences), "Should process all sequences"
        
        # All should have same tensor dimensions due to padding
        shapes = [r['attentions'].shape for r in batch_results]
        # Layers and heads should be consistent
        assert all(s[0] == shapes[0][0] for s in shapes), "Layers should be consistent"
        assert all(s[1] == shapes[0][1] for s in shapes), "Heads should be consistent"


class TestAttentionExtractionTaskModels:
    """
    Test that ALL OmniModel types support attention extraction.
    Based on the note in attention_extraction_example.py
    """

    def test_classification_model_attention(self, model_name, test_sequences):
        """Test attention extraction from classification model"""
        # Use classification model (also supports attention extraction)
        model = OmniModelForSequenceClassification(
            model=model_name,
            num_labels=2,
            trust_remote_code=True
        )
        
        sequence = test_sequences[0]
        attention_result = model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            return_on_cpu=True
        )
        
        assert "attentions" in attention_result, \
            "Classification model should support attention extraction"
        assert isinstance(attention_result['attentions'], torch.Tensor), \
            "Should return attention tensor"

    def test_regression_model_attention(self, model_name, test_sequences):
        """Test attention extraction from regression model"""
        # Use regression model (also supports attention extraction)
        model = OmniModelForSequenceRegression(
            model=model_name,
            trust_remote_code=True
        )
        
        sequence = test_sequences[0]
        attention_result = model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            return_on_cpu=True
        )
        
        assert "attentions" in attention_result, \
            "Regression model should support attention extraction"
        assert isinstance(attention_result['attentions'], torch.Tensor), \
            "Should return attention tensor"


class TestAttentionExtractionEdgeCases:
    """Test edge cases in attention extraction"""

    @pytest.fixture(scope="class")
    def embedding_model(self, model_name):
        """Load embedding model"""
        model = OmniModelForEmbedding(model=model_name, trust_remote_code=True)
        return model

    def test_very_short_sequence(self, embedding_model):
        """Test attention extraction on very short sequence"""
        sequence = "ATCG"  # Just 4 bases
        
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            return_on_cpu=True
        )
        
        assert "attentions" in attention_result, "Should handle short sequences"
        assert len(attention_result['tokens']) >= 2, "Should have at least CLS and SEP tokens"

    def test_max_length_sequence(self, embedding_model):
        """Test attention extraction on sequence at max length"""
        # Create sequence close to max length
        sequence = "ATCG" * 31  # 124 bases, will be ~126 tokens with special tokens
        
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            return_on_cpu=True
        )
        
        assert "attentions" in attention_result, "Should handle max length sequences"
        tokens_len = len(attention_result['tokens'])
        assert tokens_len <= 128, f"Token count {tokens_len} should not exceed max_length"

    def test_empty_layer_indices(self, embedding_model, test_sequences):
        """Test with empty layer indices list"""
        sequence = test_sequences[0]
        
        # Extract with empty layer list - should extract all layers
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=None,  # None means all layers
            return_on_cpu=True
        )
        
        attentions = attention_result['attentions']
        assert attentions.shape[0] > 0, "Should extract at least one layer"

    def test_invalid_layer_index_handling(self, embedding_model, test_sequences):
        """Test handling of out-of-range layer indices"""
        sequence = test_sequences[0]
        
        # Use negative index (should work - means last layer)
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=[-1],  # Last layer
            return_on_cpu=True
        )
        
        attentions = attention_result['attentions']
        assert attentions.shape[0] == 1, "Should extract exactly one layer"


@pytest.mark.slow
class TestAttentionExtractionPerformance:
    """Performance tests for attention extraction (marked as slow)"""

    @pytest.fixture(scope="class")
    def embedding_model(self, model_name):
        """Load embedding model"""
        model = OmniModelForEmbedding(model=model_name, trust_remote_code=True)
        return model

    def test_large_batch_processing(self, embedding_model):
        """Test processing a large batch of sequences"""
        # Create 20 sequences
        sequences = [f"{'ATCG' * 20}" for _ in range(20)]
        
        batch_results = embedding_model.batch_extract_attention_scores(
            sequences=sequences,
            batch_size=5,  # Process in batches of 5
            max_length=128,
            layer_indices=[0, -1],  # Only first and last to speed up
            head_indices=[0],  # Only first head
            return_on_cpu=True
        )
        
        assert len(batch_results) == 20, "Should process all 20 sequences"

    def test_all_layers_all_heads(self, embedding_model, test_sequences):
        """Test extracting all layers and all heads (most expensive operation)"""
        sequence = test_sequences[0]
        
        attention_result = embedding_model.extract_attention_scores(
            sequence=sequence,
            max_length=128,
            layer_indices=None,  # All layers
            head_indices=None,   # All heads
            return_on_cpu=True
        )
        
        attentions = attention_result['attentions']
        # Should have shape (num_layers, num_heads, seq_len, seq_len)
        assert attentions.ndim == 4, "Should be 4D tensor"
        assert attentions.shape[0] > 1, "Should have multiple layers"
        assert attentions.shape[1] > 1, "Should have multiple heads"
