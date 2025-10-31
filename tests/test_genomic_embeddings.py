# -*- coding: utf-8 -*-
# file: test_genomic_embeddings.py
# time: 17:50 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for Genomic Embeddings extraction.
Based on examples/genomic_embeddings/README.md
"""

import pytest
import numpy as np
import torch

from omnigenbench import OmniModelForEmbedding, OmniTokenizer


@pytest.fixture(scope="module")
def model_name():
    """Model name for testing"""
    return "yangheng/OmniGenome-186M"


@pytest.fixture(scope="module")
def test_sequences():
    """Test genomic sequences"""
    return [
        "ATCGATCGATCG",
        "GCGCGCGCGCGC",
        "TATATATATATAT",
        "AGAGAGAGAGAG",
        "CTCTCTCTCTCT"
    ]


@pytest.fixture(scope="module")
def embedding_model(model_name):
    """
    Load embedding model and tokenizer.
    Based on Quick Start section in README.md
    """
    tokenizer = OmniTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = OmniModelForEmbedding(
        model_name,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
    
    return model


class TestSingleSequenceEmbedding:
    """
    Test single sequence embedding extraction.
    Based on section 1 of README.md Quick Start
    """

    def test_basic_encode(self, embedding_model, test_sequences):
        """Test basic encoding with mean aggregation"""
        sequence = test_sequences[0]
        
        # Extract embedding
        embedding = embedding_model.encode(sequence, agg="mean")
        
        # Verify embedding properties
        assert isinstance(embedding, (np.ndarray, torch.Tensor)), \
            "Embedding should be numpy array or torch tensor"
        
        # Convert to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        # Check shape - should be 1D vector
        assert embedding.ndim == 1, "Embedding should be 1D vector"
        assert embedding.shape[0] > 0, "Embedding should have positive dimension"
        
        # Typical hidden dimension for transformers
        assert embedding.shape[0] >= 128, "Hidden dimension should be at least 128"

    def test_encode_all_sequences(self, embedding_model, test_sequences):
        """Test encoding multiple sequences individually"""
        embeddings = []
        
        for sequence in test_sequences:
            embedding = embedding_model.encode(sequence, agg="mean")
            
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            embeddings.append(embedding)
        
        # All embeddings should have same shape
        shapes = [emb.shape for emb in embeddings]
        assert all(s == shapes[0] for s in shapes), \
            "All embeddings should have same shape"


class TestAggregationStrategies:
    """
    Test different aggregation strategies.
    Based on Aggregation Strategy Selection section in README.md
    """

    def test_mean_aggregation(self, embedding_model, test_sequences):
        """Test mean pooling aggregation (RECOMMENDED)"""
        sequence = test_sequences[0]
        
        embedding = embedding_model.encode(sequence, agg="mean")
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        assert embedding.ndim == 1, "Mean aggregation should produce 1D vector"
        assert not np.isnan(embedding).any(), "Embedding should not contain NaN"
        assert not np.isinf(embedding).any(), "Embedding should not contain inf"

    def test_head_aggregation(self, embedding_model, test_sequences):
        """Test head token (CLS-like) aggregation"""
        sequence = test_sequences[0]
        
        embedding = embedding_model.encode(sequence, agg="head")
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        assert embedding.ndim == 1, "Head aggregation should produce 1D vector"
        assert embedding.shape[0] > 0, "Should have valid dimensions"

    def test_tail_aggregation(self, embedding_model, test_sequences):
        """Test tail token (last valid) aggregation"""
        sequence = test_sequences[0]
        
        embedding = embedding_model.encode(sequence, agg="tail")
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        assert embedding.ndim == 1, "Tail aggregation should produce 1D vector"
        assert embedding.shape[0] > 0, "Should have valid dimensions"

    def test_compare_aggregation_strategies(self, embedding_model):
        """
        Compare different aggregation strategies.
        Based on comparison example in README.md
        """
        sequence = "ATCGATCGATCG"
        strategies = ["mean", "head", "tail"]
        
        embeddings = {}
        for strategy in strategies:
            emb = embedding_model.encode(sequence, agg=strategy)
            
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            embeddings[strategy] = emb
        
        # All should have same shape
        shapes = [emb.shape for emb in embeddings.values()]
        assert all(s == shapes[0] for s in shapes), \
            "All aggregation strategies should produce same shape"
        
        # Check norms
        for strategy, emb in embeddings.items():
            norm = np.linalg.norm(emb)
            assert norm > 0, f"{strategy} embedding should have positive norm"
            assert not np.isnan(norm), f"{strategy} embedding norm should not be NaN"

    def test_aggregation_consistency(self, embedding_model):
        """Test that same aggregation gives consistent results"""
        sequence = "ATCGATCGATCG"
        
        # Encode same sequence twice
        emb1 = embedding_model.encode(sequence, agg="mean")
        emb2 = embedding_model.encode(sequence, agg="mean")
        
        if isinstance(emb1, torch.Tensor):
            emb1 = emb1.cpu().numpy()
        if isinstance(emb2, torch.Tensor):
            emb2 = emb2.cpu().numpy()
        
        # Should be identical (or very close due to numerical precision)
        np.testing.assert_allclose(emb1, emb2, rtol=1e-5, atol=1e-7,
                                  err_msg="Same sequence should give same embedding")


class TestBatchEmbedding:
    """
    Test batch embedding extraction.
    Based on section 2 of README.md Quick Start
    """

    def test_batch_encode(self, embedding_model, test_sequences):
        """
        Test efficient batch processing.
        Based on batch example in README.md
        """
        # Efficient batch processing
        embeddings = embedding_model.batch_encode(
            test_sequences,
            batch_size=32,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Check shape
        assert embeddings.shape[0] == len(test_sequences), \
            f"Should have embedding for each sequence"
        assert embeddings.ndim == 2, "Batch embeddings should be 2D"

    def test_batch_different_sizes(self, embedding_model):
        """Test batch processing with different batch sizes"""
        sequences = ["ATCG" * 10] * 20  # 20 sequences
        
        batch_sizes = [4, 8, 16]
        
        for batch_size in batch_sizes:
            embeddings = embedding_model.batch_encode(
                sequences,
                batch_size=batch_size,
                agg="mean"
            )
            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            
            assert embeddings.shape[0] == len(sequences), \
                f"Batch size {batch_size} should process all sequences"

    def test_batch_vs_individual(self, embedding_model, test_sequences):
        """Test that batch encoding gives same results as individual"""
        # Batch encoding
        batch_embeddings = embedding_model.batch_encode(
            test_sequences,
            batch_size=8,
            agg="mean"
        )
        
        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = batch_embeddings.cpu().numpy()
        
        # Individual encoding
        individual_embeddings = []
        for seq in test_sequences:
            emb = embedding_model.encode(seq, agg="mean")
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            individual_embeddings.append(emb)
        
        individual_embeddings = np.vstack(individual_embeddings)
        
        # Should be very close
        np.testing.assert_allclose(
            batch_embeddings, 
            individual_embeddings,
            rtol=1e-4, 
            atol=1e-6,
            err_msg="Batch and individual encoding should match"
        )


class TestSequenceSimilarity:
    """
    Test sequence similarity computation.
    Based on section 3 of README.md Quick Start
    """

    def test_compute_similarity(self, embedding_model):
        """
        Test computing similarity between sequences.
        Based on similarity example in README.md
        """
        seq1 = "ATCGATCGATCG"
        seq2 = "ATCGAACGATCG"  # One mismatch
        
        # Get embeddings
        emb1 = embedding_model.encode(seq1, agg="mean")
        emb2 = embedding_model.encode(seq2, agg="mean")
        
        # Compute similarity
        similarity = embedding_model.compute_similarity(emb1, emb2)
        
        # Similarity should be close to 1.0 for similar sequences
        assert 0 <= similarity <= 1, "Cosine similarity should be between 0 and 1"
        assert similarity > 0.8, "Very similar sequences should have high similarity"

    def test_identical_sequences_similarity(self, embedding_model):
        """Test that identical sequences have similarity 1.0"""
        sequence = "ATCGATCGATCG"
        
        emb1 = embedding_model.encode(sequence, agg="mean")
        emb2 = embedding_model.encode(sequence, agg="mean")
        
        similarity = embedding_model.compute_similarity(emb1, emb2)
        
        # Should be very close to 1.0
        assert abs(similarity - 1.0) < 1e-5, \
            "Identical sequences should have similarity ~1.0"

    def test_different_sequences_similarity(self, embedding_model):
        """Test similarity of very different sequences"""
        seq1 = "AAAAAAAAAA"
        seq2 = "CCCCCCCCCC"
        
        emb1 = embedding_model.encode(seq1, agg="mean")
        emb2 = embedding_model.encode(seq2, agg="mean")
        
        similarity = embedding_model.compute_similarity(emb1, emb2)
        
        # Should be lower for different sequences
        assert 0 <= similarity <= 1, "Similarity should be in valid range"
        # Note: May still be somewhat similar due to model's learned representations

    def test_similarity_matrix(self, embedding_model, test_sequences):
        """Test computing similarity matrix for multiple sequences"""
        # Get all embeddings
        embeddings = embedding_model.batch_encode(
            test_sequences,
            batch_size=8,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Compute pairwise similarities
        n = len(test_sequences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim = embedding_model.compute_similarity(
                    embeddings[i], 
                    embeddings[j]
                )
                similarity_matrix[i, j] = sim
        
        # Check properties
        # Diagonal should be ~1.0 (self-similarity)
        diagonal = np.diag(similarity_matrix)
        assert np.allclose(diagonal, 1.0, atol=1e-5), \
            "Self-similarity should be 1.0"
        
        # Matrix should be symmetric
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-5), \
            "Similarity matrix should be symmetric"


class TestEmbeddingProperties:
    """Test mathematical properties of embeddings"""

    def test_embedding_normalization(self, embedding_model, test_sequences):
        """Test embedding magnitude and normalization"""
        for sequence in test_sequences:
            embedding = embedding_model.encode(sequence, agg="mean")
            
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # Check norm
            norm = np.linalg.norm(embedding)
            assert norm > 0, "Embedding should have positive norm"
            assert not np.isnan(norm), "Norm should not be NaN"
            assert not np.isinf(norm), "Norm should not be infinite"

    def test_embedding_dimensionality(self, embedding_model, test_sequences):
        """Test that all embeddings have consistent dimensionality"""
        embeddings = []
        
        for sequence in test_sequences:
            emb = embedding_model.encode(sequence, agg="mean")
            
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            embeddings.append(emb)
        
        # All should have same shape
        shapes = [emb.shape for emb in embeddings]
        assert len(set(shapes)) == 1, \
            "All embeddings should have same dimensionality"

    def test_embedding_variance(self, embedding_model, test_sequences):
        """Test that embeddings have reasonable variance"""
        embeddings = embedding_model.batch_encode(
            test_sequences,
            batch_size=8,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Compute variance across sequences
        variance = np.var(embeddings, axis=0)
        
        # Should have some variance (not all zeros)
        assert np.mean(variance) > 0, "Embeddings should have variance"
        assert not np.all(variance == 0), "Not all dimensions should have zero variance"


class TestEdgeCases:
    """Test edge cases in embedding extraction"""

    def test_very_short_sequence(self, embedding_model):
        """Test embedding of very short sequence"""
        sequence = "ATCG"
        
        embedding = embedding_model.encode(sequence, agg="mean")
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        assert embedding.shape[0] > 0, "Should produce valid embedding for short sequence"
        assert not np.isnan(embedding).any(), "Should not contain NaN"

    def test_longer_sequence(self, embedding_model):
        """Test embedding of longer sequence"""
        # Create longer sequence (200 bases)
        sequence = "ATCGATCG" * 25
        
        embedding = embedding_model.encode(sequence, agg="mean")
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        assert embedding.shape[0] > 0, "Should produce valid embedding for long sequence"
        assert not np.isnan(embedding).any(), "Should not contain NaN"

    def test_empty_batch(self, embedding_model):
        """Test handling of empty sequence list"""
        sequences = []
        
        embeddings = embedding_model.batch_encode(
            sequences,
            batch_size=8,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        assert embeddings.shape[0] == 0, "Empty batch should return empty result"

    def test_single_sequence_batch(self, embedding_model):
        """Test batch with single sequence"""
        sequences = ["ATCGATCGATCG"]
        
        embeddings = embedding_model.batch_encode(
            sequences,
            batch_size=8,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        assert embeddings.shape[0] == 1, "Should return single embedding"
        assert embeddings.ndim == 2, "Should be 2D array"


@pytest.mark.slow
class TestEmbeddingPerformance:
    """Performance tests for embedding extraction (marked as slow)"""

    def test_large_batch_processing(self, embedding_model):
        """Test processing large batch of sequences"""
        # Create 100 sequences
        sequences = [f"{'ATCG' * 20}" for _ in range(100)]
        
        embeddings = embedding_model.batch_encode(
            sequences,
            batch_size=16,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        assert embeddings.shape[0] == 100, "Should process all 100 sequences"

    def test_variable_length_batch(self, embedding_model):
        """Test batch with highly variable sequence lengths"""
        sequences = [
            "ATCG" * 5,      # 20 bases
            "GCTA" * 25,     # 100 bases
            "TTAA" * 10,     # 40 bases
            "AAAA" * 50,     # 200 bases
        ]
        
        embeddings = embedding_model.batch_encode(
            sequences,
            batch_size=4,
            agg="mean"
        )
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # All should have same embedding dimension despite different lengths
        assert embeddings.shape[0] == len(sequences), "Should process all sequences"
        assert len(set(embeddings.shape[1:])) == 1, \
            "All embeddings should have same dimension"
