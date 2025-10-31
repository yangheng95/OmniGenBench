# -*- coding: utf-8 -*-
# file: test_structure_prediction.py
# time: 17:45 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for RNA Secondary Structure Prediction.
Based on examples/rna_secondary_structure_prediction/
"""

import pytest

from omnigenbench import ModelHub, OmniModelForTokenClassification, OmniTokenizer


@pytest.fixture(scope="module")
def test_rna_sequences():
    """Test RNA sequences for structure prediction"""
    return [
        "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA",
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
        "AGAGUGGUUGACUCGUGUGCGCGCGAGCGAGUAGCAAAGCGAGGUCGCUGGUUCGAUUCCGGCACCUCUCU"
    ]


class TestStructureValidation:
    """
    Test structure validation functions.
    Based on enhanced_ssp_demo.py
    """

    def test_validate_rna_sequence(self, test_rna_sequences):
        """Test RNA sequence validation"""
        for seq in test_rna_sequences:
            # Sequences should only contain valid RNA bases
            valid_bases = set("AUGC")
            assert all(base in valid_bases for base in seq), \
                f"Sequence should only contain A, U, G, C: {seq}"
            
            # Should not be too short
            assert len(seq) >= 10, "Sequence should be at least 10 bases"

    def test_structure_validity_loss(self):
        """
        Test calculation of structure validity loss.
        Based on ss_validity_loss() in Secondary_Structure_Prediction.py
        """
        def ss_validity_loss(rna_strct: str) -> float:
            """Calculate validity loss for RNA structure"""
            left = right = 0
            dots = rna_strct.count('.')
            for c in rna_strct:
                if c == '(':
                    left += 1
                elif c == ')':
                    if left:
                        left -= 1
                    else:
                        right += 1
                elif c != '.':
                    raise ValueError(f"Invalid char {c}")
            return (left + right) / (len(rna_strct) - dots + 1e-8)
        
        # Test valid structures (balanced)
        valid_structures = [
            "(((...)))",
            "(((..)))",
            "..(((...)))...",
            "((((....))))",
        ]
        
        for struct in valid_structures:
            loss = ss_validity_loss(struct)
            assert loss == 0.0, f"Valid structure {struct} should have loss 0.0, got {loss}"
        
        # Test invalid structures (unbalanced)
        invalid_structures = [
            "(((...",     # Unclosed opening brackets
            "...)))",     # Extra closing brackets
            "(.((",       # Mismatched - 2 opening, 0 closing
        ]
        
        for struct in invalid_structures:
            loss = ss_validity_loss(struct)
            assert loss > 0.0, f"Invalid structure {struct} should have loss > 0.0"

    def test_find_invalid_positions(self):
        """
        Test finding invalid bracket positions.
        Based on find_invalid_positions() in Secondary_Structure_Prediction.py
        """
        def find_invalid_positions(struct: str) -> list:
            """Find positions with invalid brackets"""
            stack, invalid = [], []
            for i, c in enumerate(struct):
                if c == '(':
                    stack.append(i)
                elif c == ')':
                    if stack:
                        stack.pop()
                    else:
                        invalid.append(i)
            invalid.extend(stack)
            return invalid
        
        # Valid structure - no invalid positions
        valid = "(((...)))"
        assert len(find_invalid_positions(valid)) == 0, \
            "Valid structure should have no invalid positions"
        
        # Unmatched opening brackets
        unmatched_open = "(((..."
        invalid_pos = find_invalid_positions(unmatched_open)
        assert len(invalid_pos) == 3, "Should find 3 unmatched opening brackets"
        
        # Unmatched closing brackets
        unmatched_close = "...)))"
        invalid_pos = find_invalid_positions(unmatched_close)
        assert len(invalid_pos) == 3, "Should find 3 unmatched closing brackets"


class TestBasePairExtraction:
    """
    Test base pair extraction from structures.
    Based on get_base_pairs() in enhanced_ssp_demo.py
    """

    def test_get_base_pairs(self):
        """Test extracting base pairs from dot-bracket notation"""
        def get_base_pairs(structure: str) -> set:
            """Extract base pairs from structure"""
            stack = []
            pairs = set()
            
            for i, c in enumerate(structure):
                if c == '(':
                    stack.append(i)
                elif c == ')' and stack:
                    j = stack.pop()
                    pairs.add((j, i))
            
            return pairs
        
        # Simple hairpin
        structure = "(((...)))"
        pairs = get_base_pairs(structure)
        expected_pairs = {(0, 8), (1, 7), (2, 6)}
        assert pairs == expected_pairs, f"Expected {expected_pairs}, got {pairs}"
        
        # Structure with no pairs
        structure_no_pairs = "......."
        pairs = get_base_pairs(structure_no_pairs)
        assert len(pairs) == 0, "Structure with no brackets should have no pairs"
        
        # Complex structure
        structure = "((..((...)).))"
        pairs = get_base_pairs(structure)
        # This structure has 4 pairs: (0,13), (1,12), (5,11), (6,10)
        assert len(pairs) == 4, f"Should extract all 4 base pairs, got {len(pairs)}"

    def test_base_pair_count(self):
        """Test counting base pairs in structures"""
        structures = [
            ("(((...)))", 3),      # 3 pairs
            ("(((..)))", 3),       # 3 pairs
            (".....", 0),          # 0 pairs
            ("((((....))))", 4),   # 4 pairs
        ]
        
        for struct, expected_count in structures:
            pair_count = struct.count('(')
            assert pair_count == expected_count, \
                f"Structure {struct} should have {expected_count} pairs, got {pair_count}"


class TestStructureMetrics:
    """
    Test structure comparison metrics.
    Based on calculate_structure_metrics() in enhanced_ssp_demo.py
    """

    def test_structure_accuracy(self):
        """Test calculating accuracy between predicted and true structures"""
        def calculate_accuracy(pred_struct: str, true_struct: str) -> float:
            """Calculate accuracy between two structures"""
            if len(pred_struct) != len(true_struct):
                raise ValueError("Structures must have same length")
            
            matches = sum(1 for a, b in zip(pred_struct, true_struct) if a == b)
            return matches / len(true_struct)
        
        # Identical structures
        struct1 = "(((...)))"
        struct2 = "(((...)))"
        accuracy = calculate_accuracy(struct1, struct2)
        assert accuracy == 1.0, "Identical structures should have 100% accuracy"
        
        # Completely different
        struct1 = "(((...)))"
        struct2 = "........."
        accuracy = calculate_accuracy(struct1, struct2)
        assert accuracy < 0.5, "Different structures should have low accuracy"
        
        # Partially matching
        struct1 = "(((...)))"
        struct2 = "((.(...))"  # Same length, slightly different
        accuracy = calculate_accuracy(struct1, struct2)
        assert 0 < accuracy < 1, "Partially matching should be between 0 and 1"

    def test_base_pair_precision_recall(self):
        """Test calculating precision and recall for base pairs"""
        def get_base_pairs(structure: str) -> set:
            """Extract base pairs"""
            stack = []
            pairs = set()
            for i, c in enumerate(structure):
                if c == '(':
                    stack.append(i)
                elif c == ')' and stack:
                    j = stack.pop()
                    pairs.add((j, i))
            return pairs
        
        def calculate_bp_metrics(pred_struct: str, true_struct: str) -> dict:
            """Calculate precision, recall, F1 for base pairs"""
            pred_pairs = get_base_pairs(pred_struct)
            true_pairs = get_base_pairs(true_struct)
            
            if not true_pairs:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            common_pairs = pred_pairs & true_pairs
            
            precision = len(common_pairs) / len(pred_pairs) if pred_pairs else 0
            recall = len(common_pairs) / len(true_pairs)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            return {'precision': precision, 'recall': recall, 'f1': f1}
        
        # Perfect prediction
        struct = "(((...)))"
        metrics = calculate_bp_metrics(struct, struct)
        assert metrics['precision'] == 1.0, "Perfect prediction should have precision 1.0"
        assert metrics['recall'] == 1.0, "Perfect prediction should have recall 1.0"
        assert metrics['f1'] == 1.0, "Perfect prediction should have F1 1.0"
        
        # Partial match - pred has subset of true base pairs
        pred = "(((...)))"  # Has pairs: (0,8), (1,7), (2,6)
        true = "((((.))))"  # Has pairs: (0,8), (1,7), (2,6), (3,5) - same first 3 pairs
        metrics = calculate_bp_metrics(pred, true)
        assert 0 < metrics['precision'] <= 1.0, "Precision should be between 0 and 1"
        assert 0 < metrics['recall'] <= 1.0, "Recall should be between 0 and 1"
        assert 0 < metrics['f1'] <= 1.0, "F1 should be between 0 and 1"

    def test_structure_complexity_metrics(self):
        """Test calculating structure complexity metrics"""
        structures = [
            ("(((...)))", {'base_pairs': 3, 'unpaired': 3}),
            (".....", {'base_pairs': 0, 'unpaired': 5}),
            ("((((....))))", {'base_pairs': 4, 'unpaired': 4}),
        ]
        
        for struct, expected in structures:
            base_pairs = struct.count('(')
            unpaired = struct.count('.')
            
            assert base_pairs == expected['base_pairs'], \
                f"Structure {struct} should have {expected['base_pairs']} base pairs"
            assert unpaired == expected['unpaired'], \
                f"Structure {struct} should have {expected['unpaired']} unpaired bases"


class TestModelInference:
    """Test structure prediction model inference"""

    @pytest.fixture(scope="class")
    def model(self):
        """
        Load model for structure prediction.
        Using base model - in practice would use fine-tuned SSP model.
        """
        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(
            "yangheng/OmniGenome-186M",
            trust_remote_code=True
        )
        
        # Load model with token classification head for structure prediction
        # 3 labels: '(' = paired-left, ')' = paired-right, '.' = unpaired
        model = OmniModelForTokenClassification(
            "yangheng/OmniGenome-186M",
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        model.eval()
        return model

    def test_single_sequence_prediction(self, model, test_rna_sequences):
        """Test predicting structure for single sequence"""
        sequence = test_rna_sequences[0]
        
        # Run inference
        result = model.inference(sequence)
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "predictions" in result, "Result should have predictions"
        
        # Check predictions
        predictions = result['predictions']
        assert predictions is not None, "Predictions should not be None"

    def test_batch_prediction(self, model, test_rna_sequences):
        """Test batch structure prediction"""
        # Process all sequences
        results = [model.inference(seq) for seq in test_rna_sequences]
        
        assert len(results) == len(test_rna_sequences), \
            "Should have result for each sequence"
        
        for i, result in enumerate(results):
            assert "predictions" in result, \
                f"Result {i} should have predictions"


class TestViennaRNAComparison:
    """Test comparison with ViennaRNA predictions"""

    def test_vienna_prediction(self, test_rna_sequences):
        """
        Test ViennaRNA structure prediction.
        Based on usage in Secondary_Structure_Prediction.py
        """
        pytest.importorskip("ViennaRNA", reason="ViennaRNA not available")
        
        import ViennaRNA as RNA
        
        for sequence in test_rna_sequences:
            # Predict structure using ViennaRNA
            structure, mfe = RNA.fold(sequence)
            
            # Verify output
            assert isinstance(structure, str), "Structure should be a string"
            assert len(structure) == len(sequence), \
                "Structure length should match sequence length"
            
            # Check structure is valid dot-bracket notation
            assert all(c in '().' for c in structure), \
                "Structure should only contain (, ), and ."
            
            # MFE should be negative for stable structures
            assert isinstance(mfe, float), "MFE should be a float"

    def test_vienna_vs_model_comparison(self, test_rna_sequences):
        """Test comparing ViennaRNA with model predictions"""
        pytest.importorskip("ViennaRNA", reason="ViennaRNA not available")
        
        import ViennaRNA as RNA
        
        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(
            "yangheng/OmniGenome-186M",
            trust_remote_code=True
        )
        
        # Load model with token classification head
        model = OmniModelForTokenClassification(
            "yangheng/OmniGenome-186M",
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        model.eval()
        
        sequence = test_rna_sequences[0]
        
        # ViennaRNA prediction
        vienna_struct, vienna_mfe = RNA.fold(sequence)
        
        # Model prediction
        model_result = model.inference(sequence)
        
        # Both should produce structures
        assert len(vienna_struct) == len(sequence), \
            "ViennaRNA structure should match sequence length"
        assert model_result is not None, \
            "Model should produce predictions"


class TestEdgeCases:
    """Test edge cases in structure prediction"""

    def test_very_short_sequence(self):
        """Test structure prediction on very short sequence"""
        sequence = "AUCG"
        
        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(
            "yangheng/OmniGenome-186M",
            trust_remote_code=True
        )
        
        # Load model with token classification head
        model = OmniModelForTokenClassification(
            "yangheng/OmniGenome-186M",
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        model.eval()
        
        result = model.inference(sequence)
        
        assert result is not None, "Should handle short sequences"

    def test_structure_fixing(self):
        """Test fixing invalid structures by replacing with dots"""
        def fix_invalid_structure(struct: str) -> str:
            """Fix invalid brackets by replacing with dots"""
            stack = []
            fixed = list(struct)
            
            for i, c in enumerate(struct):
                if c == '(':
                    stack.append(i)
                elif c == ')':
                    if stack:
                        stack.pop()
                    else:
                        fixed[i] = '.'
            
            # Fix unmatched opening brackets
            for i in stack:
                fixed[i] = '.'
            
            return ''.join(fixed)
        
        # Invalid structure
        invalid = "(((...))"
        fixed = fix_invalid_structure(invalid)
        
        # Check fixed structure is valid
        left = right = 0
        for c in fixed:
            if c == '(':
                left += 1
            elif c == ')':
                if left:
                    left -= 1
                else:
                    right += 1
        
        assert left == 0 and right == 0, \
            f"Fixed structure {fixed} should be balanced"


@pytest.mark.slow
class TestStructurePredictionPerformance:
    """Performance tests for structure prediction (marked as slow)"""

    def test_long_sequence_prediction(self):
        """Test prediction on longer RNA sequence"""
        # tRNA-like sequence (76 bases)
        long_sequence = "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA"
        
        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(
            "yangheng/OmniGenome-186M",
            trust_remote_code=True
        )
        
        # Load model with token classification head
        model = OmniModelForTokenClassification(
            "yangheng/OmniGenome-186M",
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        model.eval()
        
        result = model.inference(long_sequence)
        
        assert result is not None, "Should handle longer sequences"

    def test_batch_performance(self, test_rna_sequences):
        """Test processing multiple sequences"""
        # Create larger batch
        sequences = test_rna_sequences * 5  # 15 sequences
        
        # Load tokenizer
        tokenizer = OmniTokenizer.from_pretrained(
            "yangheng/OmniGenome-186M",
            trust_remote_code=True
        )
        
        # Load model with token classification head
        model = OmniModelForTokenClassification(
            "yangheng/OmniGenome-186M",
            tokenizer=tokenizer,
            num_labels=3,
            trust_remote_code=True
        )
        model.eval()
        
        results = [model.inference(seq) for seq in sequences]
        
        assert len(results) == len(sequences), \
            "Should process all sequences"
