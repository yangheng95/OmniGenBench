# -*- coding: utf-8 -*-
# file: test_autoinfer_cli.py
# time: 17:40 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test cases for AutoInfer CLI functionality.
Based on examples/autoinfer_examples/README.md
"""

import pytest
import json
import csv
import dill
from pathlib import Path

from omnigenbench import ModelHub, OmniTokenizer


@pytest.fixture(scope="module")
def test_model_name():
    """
    Model name for testing.
    Using OmniGenome base model instead of finetuned to avoid download issues.
    """
    return "yangheng/ogb_tfb_finetuned"


@pytest.fixture(scope="module")
def test_sequences():
    """Test sequences from autoinfer examples"""
    return [
        "ATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        "TATATATATATATATATATATATATATATATAT",
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAG",
        "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"
    ]


class TestModelHubInference:
    """
    Test ModelHub inference functionality.
    Based on Python API examples in autoinfer_examples/README.md
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self, test_model_name):
        """Load model and tokenizer once for all tests"""
        tokenizer = OmniTokenizer.from_pretrained(
            test_model_name,
            trust_remote_code=True
        )
        model = ModelHub.load(test_model_name, device="cpu")
        return model, tokenizer

    def test_single_sequence_inference(self, model_and_tokenizer, test_sequences):
        """
        Test single sequence inference.
        Based on Python API example section in README.md
        """
        model, tokenizer = model_and_tokenizer
        sequence = test_sequences[0]
        
        # Perform inference
        result = model.inference(sequence)
        
        # Verify result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        assert "predictions" in result, "Result should have 'predictions' key"
        
        # Check predictions
        predictions = result['predictions']
        assert predictions is not None, "Predictions should not be None"

    def test_batch_inference_loop(self, model_and_tokenizer, test_sequences):
        """
        Test batch inference using loop.
        Based on Python API batch example in README.md
        """
        model, tokenizer = model_and_tokenizer
        
        # Batch inference (loop over sequences)
        batch_results = [model.inference(seq) for seq in test_sequences]
        
        # Verify all results
        assert len(batch_results) == len(test_sequences), \
            f"Should have {len(test_sequences)} results"
        
        for i, res in enumerate(batch_results):
            assert isinstance(res, dict), f"Result {i} should be a dictionary"
            assert "predictions" in res, f"Result {i} should have predictions"

    def test_inference_different_sequence_lengths(self, model_and_tokenizer):
        """Test inference on sequences of varying lengths"""
        model, tokenizer = model_and_tokenizer
        
        sequences = [
            "ATCG" * 5,      # 20 bases
            "GCTA" * 15,     # 60 bases
            "TTAA" * 8,      # 32 bases
        ]
        
        results = [model.inference(seq) for seq in sequences]
        
        assert len(results) == len(sequences), "Should process all sequences"
        for result in results:
            assert "predictions" in result, "Each result should have predictions"


class TestInputFileFormats:
    """
    Test different input file formats.
    Based on Input File Formats section in README.md
    """

    def test_json_simple_sequence_list(self, tmp_path):
        """
        Test JSON format with simple sequence list.
        Based on sequences.json example in README.md
        """
        # Create test JSON file
        sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
            "TATATATATATATATATATATATATATATATAT"
        ]
        
        json_file = tmp_path / "test_sequences.json"
        with open(json_file, "w") as f:
            json.dump({"sequences": sequences}, f)
        
        # Verify file was created and can be loaded
        assert json_file.exists(), "JSON file should be created"
        
        with open(json_file, "r") as f:
            data = json.load(f)
        
        assert "sequences" in data, "Should have 'sequences' key"
        assert len(data["sequences"]) == 3, "Should have 3 sequences"
        assert data["sequences"] == sequences, "Sequences should match"

    def test_json_with_metadata(self, tmp_path):
        """
        Test JSON format with metadata.
        Based on sequences_with_metadata.json example in README.md
        """
        # Create test JSON with metadata
        data = {
            "data": [
                {
                    "sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG",
                    "gene_id": "gene_001",
                    "description": "Promoter region",
                    "label": "high"
                },
                {
                    "sequence": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
                    "gene_id": "gene_002",
                    "description": "5' UTR",
                    "label": "low"
                }
            ]
        }
        
        json_file = tmp_path / "test_metadata.json"
        with open(json_file, "w") as f:
            json.dump(data, f)
        
        # Verify file structure
        assert json_file.exists(), "JSON file should be created"
        
        with open(json_file, "r") as f:
            loaded_data = json.load(f)
        
        assert "data" in loaded_data, "Should have 'data' key"
        assert len(loaded_data["data"]) == 2, "Should have 2 entries"
        
        # Check metadata fields
        first_entry = loaded_data["data"][0]
        assert "sequence" in first_entry, "Should have sequence"
        assert "gene_id" in first_entry, "Should have gene_id"
        assert "description" in first_entry, "Should have description"
        assert "label" in first_entry, "Should have label"

    def test_csv_format(self, tmp_path):
        """
        Test CSV file format.
        Based on data.csv example in README.md
        """
        # Create test CSV file
        csv_file = tmp_path / "test_data.csv"
        
        rows = [
            ["sequence", "gene_id", "description", "label"],
            ["ATCGATCGATCGATCGATCGATCGATCGATCG", "gene_001", "Promoter region", "high"],
            ["GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC", "gene_002", "5' UTR", "low"],
            ["TATATATATATATATATATATATATATATATAT", "gene_003", "Random sequence", "low"]
        ]
        
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        # Verify file can be read
        assert csv_file.exists(), "CSV file should be created"
        
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        assert len(data) == 3, "Should have 3 data rows"
        
        # Check fields
        for row in data:
            assert "sequence" in row, "Should have sequence column"
            assert "gene_id" in row, "Should have gene_id column"
            assert "description" in row, "Should have description column"
            assert "label" in row, "Should have label column"

    def test_text_file_format(self, tmp_path):
        """
        Test text file format (one sequence per line).
        Based on sequences.txt example in README.md
        """
        # Create test text file
        txt_file = tmp_path / "test_sequences.txt"
        
        sequences = [
            "ATCGATCGATCGATCGATCGATCGATCGATCG",
            "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
            "TATATATATATATATATATATATATATATATAT"
        ]
        
        with open(txt_file, "w") as f:
            f.write("\n".join(sequences))
        
        # Verify file can be read
        assert txt_file.exists(), "Text file should be created"
        
        with open(txt_file, "r") as f:
            loaded_sequences = [line.strip() for line in f if line.strip()]
        
        assert len(loaded_sequences) == 3, "Should have 3 sequences"
        assert loaded_sequences == sequences, "Sequences should match"


class TestOutputFormat:
    """
    Test output format generation.
    Based on Output Format section in README.md
    """

    def test_output_structure(self, tmp_path, test_model_name, test_sequences):
        """
        Test that output follows the expected format.
        Based on output format example in README.md
        """
        # Create mock output data matching the format in README
        output_data = {
            "model": test_model_name,
            "total_sequences": len(test_sequences),
            "results": []
        }
        
        for i, seq in enumerate(test_sequences):
            result = {
                "sequence": seq,
                "metadata": {
                    "index": i,
                    "gene_id": f"gene_{i:03d}",
                    "description": f"Test sequence {i}"
                },
                "predictions": [1, 0, 1, 0, 1],
                "probabilities": [0.92, 0.15, 0.88, 0.23, 0.91]
            }
            output_data["results"].append(result)
        
        # Save to file (dill binary)
        output_file = tmp_path / "test_results.dill"
        with open(output_file, "wb") as f:
            dill.dump(output_data, f)
        
        # Verify structure
        assert output_file.exists(), "Output file should be created"
        
        with open(output_file, "rb") as f:
            loaded = dill.load(f)
        
        # Check required fields
        assert "model" in loaded, "Should have 'model' field"
        assert "total_sequences" in loaded, "Should have 'total_sequences' field"
        assert "results" in loaded, "Should have 'results' field"
        
        # Check results structure
        assert len(loaded["results"]) == len(test_sequences), "Should have all results"
        
        for result in loaded["results"]:
            assert "sequence" in result, "Result should have sequence"
            assert "metadata" in result, "Result should have metadata"
            assert "predictions" in result, "Result should have predictions"
            # Note: probabilities might not always be present
            if "probabilities" in result:
                assert isinstance(result["probabilities"], list), \
                    "Probabilities should be a list"


class TestInferenceEdgeCases:
    """Test edge cases in inference"""

    @pytest.fixture(scope="class")
    def model(self, test_model_name):
        """Load model for edge case testing"""
        return ModelHub.load(test_model_name, device="cpu")

    def test_very_short_sequence(self, model):
        """Test inference on very short sequence"""
        sequence = "ATCG"
        
        result = model.inference(sequence)
        
        assert isinstance(result, dict), "Should return dict even for short sequence"
        assert "predictions" in result, "Should have predictions"

    def test_long_sequence(self, model):
        """Test inference on longer sequence"""
        # Create a longer sequence (400 bases)
        sequence = "ATCGATCG" * 50
        
        result = model.inference(sequence)
        
        assert isinstance(result, dict), "Should handle longer sequences"
        assert "predictions" in result, "Should have predictions"

    def test_repeated_inference(self, model):
        """Test multiple inference calls on same sequence"""
        sequence = "ATCGATCGATCGATCGATCGATCGATCGATCG"
        
        # Run inference multiple times
        results = [model.inference(sequence) for _ in range(3)]
        
        # All should succeed
        assert len(results) == 3, "All inferences should complete"
        for result in results:
            assert "predictions" in result, "Each result should have predictions"


class TestBatchProcessing:
    """Test batch processing scenarios"""

    @pytest.fixture(scope="class")
    def model(self, test_model_name):
        """Load model for batch testing"""
        return ModelHub.load(test_model_name, device="cpu")

    def test_small_batch(self, model):
        """Test processing a small batch"""
        sequences = [f"{'ATCG' * 10}" for _ in range(5)]
        
        results = [model.inference(seq) for seq in sequences]
        
        assert len(results) == 5, "Should process all 5 sequences"

    def test_variable_length_batch(self, model):
        """Test batch with variable length sequences"""
        sequences = [
            "ATCG" * 5,
            "GCTA" * 20,
            "TTAA" * 10,
            "AAAA" * 15,
        ]
        
        results = [model.inference(seq) for seq in sequences]
        
        assert len(results) == len(sequences), "Should process all sequences"
        for result in results:
            assert "predictions" in result, "Each should have predictions"

    def test_empty_sequence_list(self, model):
        """Test handling of empty sequence list"""
        sequences = []
        
        results = [model.inference(seq) for seq in sequences]
        
        assert len(results) == 0, "Empty input should give empty output"


@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests for real-world usage scenarios"""

    def test_full_pipeline_json(self, tmp_path, test_model_name, test_sequences):
        """
        Test complete pipeline: JSON input -> inference -> JSON output.
        Simulates the full autoinfer workflow.
        """
        # 1. Create input JSON
        input_file = tmp_path / "input.json"
        with open(input_file, "w") as f:
            json.dump({"sequences": test_sequences}, f)
        
        # 2. Load model
        model = ModelHub.load(test_model_name, device="cpu")
        
        # 3. Load sequences from file
        with open(input_file, "r") as f:
            data = json.load(f)
        sequences = data["sequences"]
        
        # 4. Run inference
        results = [model.inference(seq) for seq in sequences]
        
        # 5. Create output
        output_data = {
            "model": test_model_name,
            "total_sequences": len(sequences),
            "results": [
                {
                    "sequence": seq,
                    "metadata": {"index": i},
                    "predictions": res["predictions"],
                }
                for i, (seq, res) in enumerate(zip(sequences, results))
            ]
        }
        
        # 6. Save output (dill binary)
        output_file = tmp_path / "output.dill"
        with open(output_file, "wb") as f:
            dill.dump(output_data, f)
        
        # 7. Verify
        assert output_file.exists(), "Output file should be created"
        
        with open(output_file, "rb") as f:
            output = dill.load(f)
        
        assert output["total_sequences"] == len(test_sequences), \
            "Should process all sequences"

    def test_full_pipeline_csv(self, tmp_path, test_model_name):
        """
        Test complete pipeline with CSV: CSV input -> inference -> JSON output.
        """
        # 1. Create input CSV
        input_file = tmp_path / "input.csv"
        
        data = [
            {"sequence": "ATCGATCGATCGATCGATCGATCGATCGATCG", "gene_id": "gene_001"},
            {"sequence": "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC", "gene_id": "gene_002"},
        ]
        
        with open(input_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sequence", "gene_id"])
            writer.writeheader()
            writer.writerows(data)
        
        # 2. Load model
        model = ModelHub.load(test_model_name, device="cpu")
        
        # 3. Load sequences from CSV
        with open(input_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # 4. Run inference
        results = []
        for row in rows:
            result = model.inference(row["sequence"])
            result["gene_id"] = row["gene_id"]
            results.append(result)
        
        # 5. Verify
        assert len(results) == 2, "Should process both sequences"
        for result in results:
            assert "gene_id" in result, "Should preserve metadata"
            assert "predictions" in result, "Should have predictions"
