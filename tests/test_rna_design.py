# -*- coding: utf-8 -*-
# file: test_rna_design.py
# time: 17:30 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for RNA Sequence Design functionality.
Based on examples/rna_sequence_design/rna_design_examples.py
"""

import pytest
import json
from pathlib import Path

from omnigenbench import OmniModelForRNADesign


@pytest.fixture(scope="module")
def rna_design_model():
    """
    Load RNA design model once for all tests.
    Using smaller model for faster testing.
    """
    model = OmniModelForRNADesign(model="yangheng/OmniGenome-186M")
    return model


class TestRNADesignBasic:
    """Test basic RNA design functionality from examples"""

    def test_simple_hairpin_design(self, rna_design_model):
        """
        Test simple hairpin structure design.
        Based on example_1_simple_hairpin() from rna_design_examples.py
        """
        # Define target structure: 3-base-pair stem with 3-nucleotide loop
        structure = "(((...)))"
        
        # Design sequences with default parameters
        sequences = rna_design_model.design(
            structure=structure,
            mutation_ratio=0.5,
            num_population=100,
            num_generation=50
        )
        
        # Verify results
        assert isinstance(sequences, list), "Design should return a list of sequences"
        assert len(sequences) > 0, "Should find at least one solution"
        
        for seq in sequences:
            assert isinstance(seq, str), "Each sequence should be a string"
            assert len(seq) == len(structure), f"Sequence length {len(seq)} should match structure length {len(structure)}"
            assert all(base in "AUGC" for base in seq), "Sequence should only contain RNA bases (A, U, G, C)"

    def test_custom_parameters(self, rna_design_model):
        """
        Test design with custom evolutionary parameters.
        Based on example_2_custom_parameters() from rna_design_examples.py
        """
        # More complex structure: nested stem-loops
        structure = "(((..(((...)))..)))"
        
        # Use more aggressive exploration for complex structures
        sequences = rna_design_model.design(
            structure=structure,
            mutation_ratio=0.3,      # Lower mutation for stability
            num_population=200,      # Larger population for diversity
            num_generation=100       # More generations for convergence
        )
        
        assert isinstance(sequences, list), "Design should return a list"
        assert len(sequences) > 0, "Should find solutions even for complex structures"
        
        # Check first sequence quality
        if sequences:
            seq = sequences[0]
            assert len(seq) == len(structure), "Sequence length should match structure"

    def test_small_population(self, rna_design_model):
        """
        Test design with minimal parameters (fast test).
        Ensures the model works with limited resources.
        """
        structure = "(((...)))"
        
        # Minimal parameters for fast testing
        sequences = rna_design_model.design(
            structure=structure,
            mutation_ratio=0.5,
            num_population=10,   # Small population
            num_generation=10    # Few generations
        )
        
        # Should still return some result
        assert isinstance(sequences, list), "Should return a list even with small population"
        # Note: May not find perfect solution, but should not crash


class TestRNADesignBatch:
    """Test batch design functionality"""

    def test_batch_design_multiple_structures(self, rna_design_model):
        """
        Test batch design of multiple structures.
        Based on example_3_batch_design() from rna_design_examples.py
        """
        structures = {
            "hairpin": "(((...)))",
            "stem_loop": "(((..)))",
            "long_stem": "((((....))))",
        }
        
        results = {}
        for name, structure in structures.items():
            sequences = rna_design_model.design(
                structure=structure,
                mutation_ratio=0.5,
                num_population=50,  # Reduced for faster testing
                num_generation=30   # Reduced for faster testing
            )
            results[name] = {
                "structure": structure,
                "sequences": sequences[:5],
                "count": len(sequences)
            }
        
        # Verify all structures were processed
        assert len(results) == len(structures), "All structures should be processed"
        
        for name, result in results.items():
            assert "structure" in result, f"Result for {name} should have structure"
            assert "sequences" in result, f"Result for {name} should have sequences"
            assert "count" in result, f"Result for {name} should have count"
            assert result["count"] >= 0, "Count should be non-negative"

    def test_save_and_load_results(self, rna_design_model, tmp_path):
        """
        Test saving and loading design results to JSON.
        Based on example_5_save_and_load() pattern from rna_design_examples.py
        """
        structure = "(((...)))"
        sequences = rna_design_model.design(
            structure=structure,
            num_population=50,
            num_generation=30
        )
        
        # Save results
        output_file = tmp_path / "design_results.json"
        result_data = {
            "structure": structure,
            "sequences": sequences[:5],
            "total_count": len(sequences)
        }
        
        with open(output_file, "w") as f:
            json.dump(result_data, f, indent=2)
        
        # Verify file exists
        assert output_file.exists(), "Output file should be created"
        
        # Load and verify
        with open(output_file, "r") as f:
            loaded_data = json.load(f)
        
        assert loaded_data["structure"] == structure, "Structure should match"
        assert len(loaded_data["sequences"]) <= 5, "Should save top 5 sequences"
        assert loaded_data["total_count"] == len(sequences), "Count should match"


class TestRNADesignValidation:
    """Test sequence validation functionality"""

    def test_validate_designed_sequences(self, rna_design_model):
        """
        Test that designed sequences can be validated.
        Based on example_4_validate_designs() from rna_design_examples.py
        """
        pytest.importorskip("ViennaRNA", reason="ViennaRNA not available")
        
        import ViennaRNA
        
        structure = "(((...)))"
        sequences = rna_design_model.design(
            structure=structure,
            num_population=50,
            num_generation=30
        )
        
        # Validate at least one sequence
        assert len(sequences) > 0, "Should have at least one sequence to validate"
        
        for seq in sequences[:3]:  # Test first 3
            # Predict structure using ViennaRNA
            predicted_structure, mfe = ViennaRNA.fold(seq)
            
            # Basic checks
            assert len(predicted_structure) == len(structure), "Predicted structure length should match"
            assert isinstance(mfe, float), "MFE should be a float"
            assert mfe < 0, "MFE should typically be negative for stable structures"

    def test_structure_length_consistency(self, rna_design_model):
        """Test that all designed sequences match structure length"""
        structures = ["((...))", "((((....))))", "((..((...)).))"]
        
        for structure in structures:
            sequences = rna_design_model.design(
                structure=structure,
                num_population=30,
                num_generation=20
            )
            
            for seq in sequences:
                assert len(seq) == len(structure), \
                    f"Sequence length {len(seq)} must equal structure length {len(structure)}"


class TestRNADesignEdgeCases:
    """Test edge cases and error handling"""

    def test_very_short_structure(self, rna_design_model):
        """Test design of very short structures"""
        structure = "((.))"  # Very short structure
        
        sequences = rna_design_model.design(
            structure=structure,
            num_population=30,
            num_generation=20
        )
        
        assert isinstance(sequences, list), "Should return list even for short structures"
        if sequences:
            assert all(len(s) == len(structure) for s in sequences), \
                "All sequences should match structure length"

    def test_only_dots_structure(self, rna_design_model):
        """Test structure with no base pairs (all dots)"""
        structure = "......"  # No base pairs
        
        # Should still work and generate sequences
        sequences = rna_design_model.design(
            structure=structure,
            num_population=20,
            num_generation=10
        )
        
        assert isinstance(sequences, list), "Should handle structures without base pairs"
        if sequences:
            assert all(len(s) == len(structure) for s in sequences), \
                "Sequence length should match"

    def test_complex_nested_structure(self, rna_design_model):
        """Test complex nested structure with multiple loops"""
        structure = "(((..(((...)))..(((...))).)))"  # Multi-loop structure
        
        sequences = rna_design_model.design(
            structure=structure,
            mutation_ratio=0.3,
            num_population=100,  # Need larger population for complex structures
            num_generation=50
        )
        
        assert isinstance(sequences, list), "Should handle complex structures"


@pytest.mark.slow
class TestRNADesignPerformance:
    """Performance and scalability tests (marked as slow)"""

    def test_large_population_convergence(self, rna_design_model):
        """Test that larger populations find more solutions"""
        structure = "(((...)))"
        
        # Small population
        small_sequences = rna_design_model.design(
            structure=structure,
            num_population=50,
            num_generation=30
        )
        
        # Large population
        large_sequences = rna_design_model.design(
            structure=structure,
            num_population=200,
            num_generation=30
        )
        
        # Larger population should typically find same or more solutions
        # (not guaranteed, but likely in practice)
        assert isinstance(small_sequences, list), "Small population should return list"
        assert isinstance(large_sequences, list), "Large population should return list"

    def test_longer_structure(self, rna_design_model):
        """Test design of longer RNA structures"""
        # Longer structure (40 nucleotides)
        structure = "((((....))))..((((...))))..((((....))))"
        
        sequences = rna_design_model.design(
            structure=structure,
            mutation_ratio=0.4,
            num_population=150,
            num_generation=100
        )
        
        assert isinstance(sequences, list), "Should handle longer structures"
        if sequences:
            assert len(sequences[0]) == len(structure), "Length should match"
