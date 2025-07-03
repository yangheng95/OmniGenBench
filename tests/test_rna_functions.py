"""
Test RNA-specific functionality based on examples.
"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock


class TestRNAFunctions:
    """Test RNA functionality based on examples."""

    def test_rna_sequence_validity_checker(self):
        """Test ss_validity_loss function from Secondary_Structure_Prediction.py."""
        # Recreate the function from the example
        def ss_validity_loss(rna_strct: str) -> float:
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

        # Test valid structures
        assert ss_validity_loss("(())") == 0.0
        assert ss_validity_loss("((..))") == 0.0
        assert ss_validity_loss("....") == 0.0
        
        # Test invalid structures
        assert ss_validity_loss("(((") > 0.0  # Unmatched left
        assert ss_validity_loss(")))") > 0.0  # Unmatched right
        assert ss_validity_loss("())(") > 0.0  # Mixed unmatched

        # Test error case
        with pytest.raises(ValueError, match="Invalid char"):
            ss_validity_loss("((X))")

    def test_find_invalid_positions(self):
        """Test find_invalid_positions function from Secondary_Structure_Prediction.py."""
        # Recreate the function from the example
        def find_invalid_positions(struct: str) -> list:
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

        # Test valid structures
        assert find_invalid_positions("(())") == []
        assert find_invalid_positions("((..))") == []
        assert find_invalid_positions("....") == []

        # Test invalid structures
        assert find_invalid_positions("(((") == [0, 1, 2]  # All unmatched left
        assert find_invalid_positions(")))") == [0, 1, 2]  # All unmatched right
        assert find_invalid_positions("())(") == [2, 3]   # One unmatched right, one left

    def test_rna_structure_formats(self):
        """Test RNA structure format validation."""
        valid_structures = [
            "(())",
            "((()))",
            ".((.))",
            "....",
            "",
            "((..))",
        ]
        
        invalid_structures = [
            "((X))",  # Invalid character
            "(()",    # Unmatched
            "())",    # Unmatched
            ")(",     # Wrong order
        ]
        
        def is_valid_structure_format(struct: str) -> bool:
            """Check if structure contains only valid characters."""
            return all(c in "()." for c in struct)
        
        for struct in valid_structures:
            assert is_valid_structure_format(struct), f"Should be valid: {struct}"
            
        for struct in invalid_structures:
            if any(c not in "()." for c in struct):
                assert not is_valid_structure_format(struct), f"Should be invalid: {struct}"

    def test_sequence_replacement_patterns(self):
        """Test U/T replacement patterns from examples."""
        # Pattern from web_rna_design.py
        def rna_to_dna_pattern(sequence):
            return sequence.replace("U", "T")
        
        def dna_to_rna_pattern(sequence):
            return sequence.replace("T", "U")
        
        # Test RNA to DNA
        assert rna_to_dna_pattern("AUCG") == "ATCG"
        assert rna_to_dna_pattern("UUUU") == "TTTT"
        assert rna_to_dna_pattern("ACGU") == "ACGT"
        
        # Test DNA to RNA
        assert dna_to_rna_pattern("ATCG") == "AUCG"
        assert dna_to_rna_pattern("TTTT") == "UUUU"
        assert dna_to_rna_pattern("ACGT") == "ACGU"

    def test_random_base_generation_patterns(self):
        """Test random base generation patterns from RNA design examples."""
        import random
        
        def generate_random_rna_base():
            """Pattern from easy_rna_design_emoo.py."""
            return random.choice(["A", "U", "G", "C"])
        
        def generate_random_dna_base():
            """Pattern from easy_rna_design_emoo.py."""
            return random.choice(["A", "T", "G", "C"])
        
        # Test multiple generations to ensure valid bases
        for _ in range(10):
            rna_base = generate_random_rna_base()
            assert rna_base in ["A", "U", "G", "C"]
            
            dna_base = generate_random_dna_base()
            assert dna_base in ["A", "T", "G", "C"]

    def test_sequence_mutation_pattern(self):
        """Test sequence mutation pattern from mlm_mutate function."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        def mutate_sequence_pattern(sequence, mutation_rate=0.1):
            """Simplified version of mutation pattern from examples."""
            sequence_array = np.array(list(sequence), dtype=np.str_)
            probability_matrix = np.full(sequence_array.shape, mutation_rate)
            masked_indices = np.random.rand(*sequence_array.shape) < probability_matrix
            sequence_array[masked_indices] = "$"  # Mask token
            return "".join(sequence_array.tolist())
        
        # Test mutation with 0% rate
        original = "AUCG"
        mutated_zero = mutate_sequence_pattern(original, 0.0)
        assert mutated_zero == original
        
        # Test mutation with 100% rate
        mutated_full = mutate_sequence_pattern(original, 1.0)
        assert mutated_full == "$$$$"
        
        # Test with moderate rate - should have some masks
        np.random.seed(42)  # For reproducible test
        mutated_partial = mutate_sequence_pattern("AUCGAUCGAUCG", 0.5)
        assert "$" in mutated_partial

    @patch('tempfile.mkdtemp')
    def test_temp_directory_pattern(self, mock_mkdtemp):
        """Test temp directory usage pattern from Secondary_Structure_Prediction.py."""
        from pathlib import Path
        
        mock_mkdtemp.return_value = "/tmp/test_dir"
        
        # Pattern from Secondary_Structure_Prediction.py
        TEMP_DIR = Path(tempfile.mkdtemp())
        
        mock_mkdtemp.assert_called_once()
        assert isinstance(TEMP_DIR, Path)

    def test_rna_embedding_sequence_validation(self):
        """Test RNA sequence validation for embedding examples."""
        # RNA sequences from RNA_Embedding_Tutorial.ipynb
        rna_sequences = [
            "AUGGCUACG",
            "CGGAUACGGC",
            "UGGCCAAGUC",
            "AUGCUGCUAUGCUA"
        ]
        
        def validate_rna_sequence(seq):
            """Validate RNA sequence format."""
            return all(base in "AUCG" for base in seq) and len(seq) > 0
        
        for seq in rna_sequences:
            assert validate_rna_sequence(seq), f"Invalid RNA sequence: {seq}"

    def test_structure_prediction_mock_pattern(self):
        """Test structure prediction pattern without ViennaRNA dependency."""
        def mock_predict_structure_single(sequence):
            """Mock version of predict_structure_single from examples."""
            # Return a mock structure and energy
            return "." * len(sequence), -10.0
        
        # Test the pattern
        seq = "AUCG"
        struct, energy = mock_predict_structure_single(seq)
        
        assert len(struct) == len(seq)
        assert isinstance(energy, float)
        assert struct == "...."

    def test_base64_encoding_pattern(self):
        """Test base64 encoding pattern from SVG generation."""
        import base64
        
        def create_mock_svg_datauri(content="test"):
            """Mock version of SVG data URI creation."""
            svg_content = f'<svg>{content}</svg>'
            b64 = base64.b64encode(svg_content.encode()).decode('utf-8')
            return f"data:image/svg+xml;base64,{b64}"
        
        uri = create_mock_svg_datauri("test")
        assert uri.startswith("data:image/svg+xml;base64,")
        
        # Decode and verify
        _, b64_part = uri.split(",", 1)
        decoded = base64.b64decode(b64_part).decode('utf-8')
        assert decoded == "<svg>test</svg>"

    def test_longest_bp_span_function(self):
        """Test longest_bp_span function from easy_rna_design_emoo.py."""
        def longest_bp_span(structure):
            """Function from easy_rna_design_emoo.py."""
            stack = []
            max_span = 0

            for i, char in enumerate(structure):
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if stack:
                        left_index = stack.pop()
                        current_span = i - left_index
                        max_span = max(max_span, current_span)

            return max_span

        # Test cases
        assert longest_bp_span("(())") == 3  # Outer pair spans 3 positions
        assert longest_bp_span("((()))") == 5  # Outer pair spans 5 positions
        assert longest_bp_span("()()") == 1   # Each pair spans 1 position
        assert longest_bp_span("....") == 0   # No pairs
        assert longest_bp_span("") == 0       # Empty structure
        assert longest_bp_span("((.))") == 4  # Outer pair spans 4 positions 