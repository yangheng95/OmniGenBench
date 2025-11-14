# -*- coding: utf-8 -*-
# file: test_te_bpp_model.py
# time: 12:30 01/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test script for TE-BPP model implementation.

This script validates the core functionality of the TEModelWithBPP:
1. Dataset creation and BPP computation
2. Model initialization
3. Forward pass
4. Training pipeline
5. Inference

Run: python test_te_bpp_model.py
"""

import os
import sys
import torch
import numpy as np
from typing import Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from te_bpp_model import (
    TEDatasetWithBPP,
    TEModelWithBPP,
    BPPProcessor,
    FeatureFusion,
    create_demo_dataset,
)

from omnigenbench import OmniTokenizer


def test_bpp_computation():
    """Test BPP matrix computation."""
    print("\n" + "=" * 80)
    print("TEST 1: BPP Matrix Computation")
    print("=" * 80)
    
    # Create a simple RNA sequence
    sequence = "AUGCAUGCGCGCGCGCAUGCAUGC"
    
    print(f"[INFO] Test sequence: {sequence}")
    print(f"[INFO] Length: {len(sequence)}")
    
    # Initialize dataset to access BPP computation method
    tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
    dataset = TEDatasetWithBPP(
        dataset_name_or_path=[],  # Empty dataset for testing
        tokenizer=tokenizer,
        max_length=512
    )
    
    # Compute BPP
    bpp_matrix = dataset.compute_bpp_matrix(sequence)
    
    # Validate BPP matrix
    print(f"\n[CHECK] BPP matrix shape: {bpp_matrix.shape}")
    assert bpp_matrix.shape == (len(sequence), len(sequence)), "BPP shape mismatch"
    
    print(f"[CHECK] BPP matrix dtype: {bpp_matrix.dtype}")
    assert bpp_matrix.dtype == np.float32, "BPP dtype should be float32"
    
    print(f"[CHECK] BPP symmetry: ", end="")
    is_symmetric = np.allclose(bpp_matrix, bpp_matrix.T)
    assert is_symmetric, "BPP matrix should be symmetric"
    print("PASS")
    
    print(f"[CHECK] BPP value range: [{bpp_matrix.min():.4f}, {bpp_matrix.max():.4f}]")
    assert bpp_matrix.min() >= 0 and bpp_matrix.max() <= 1, "BPP values out of range"
    
    print(f"[CHECK] Diagonal elements: {np.diag(bpp_matrix).sum()}")
    assert np.allclose(np.diag(bpp_matrix), 0), "Diagonal should be zero"
    
    print(f"\n[SUCCESS] BPP computation test passed!")
    return True


def test_dataset_preparation():
    """Test dataset with BPP feature preparation."""
    print("\n" + "=" * 80)
    print("TEST 2: Dataset Preparation")
    print("=" * 80)
    
    # Create demo dataset
    dataset_dir = create_demo_dataset("test_te_dataset")
    
    # Load dataset
    tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
    label2id = {"0": 0, "1": 1}
    
    dataset = TEDatasetWithBPP(
        dataset_name_or_path=f"{dataset_dir}/train.json",
        tokenizer=tokenizer,
        max_length=512,
        label2id=label2id
    )
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\n[CHECK] Sample keys: {list(sample.keys())}")
    required_keys = ["input_ids", "attention_mask", "bpp_matrix", "labels"]
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    print(f"[CHECK] input_ids shape: {sample['input_ids'].shape}")
    print(f"[CHECK] attention_mask shape: {sample['attention_mask'].shape}")
    print(f"[CHECK] bpp_matrix shape: {sample['bpp_matrix'].shape}")
    print(f"[CHECK] labels shape: {sample['labels'].shape}")
    
    assert sample['bpp_matrix'].shape == (512, 512), "BPP matrix should be 512x512"
    assert sample['input_ids'].shape[0] == 512, "Input should be padded to 512"
    
    print(f"\n[SUCCESS] Dataset preparation test passed!")
    
    # Cleanup
    import shutil
    shutil.rmtree(dataset_dir)
    
    return True


def test_bpp_processor():
    """Test BPP processor module."""
    print("\n" + "=" * 80)
    print("TEST 3: BPP Processor")
    print("=" * 80)
    
    # Create processor
    processor = BPPProcessor(output_dim=128)
    
    # Create dummy BPP matrix
    batch_size = 4
    seq_len = 512
    bpp_matrix = torch.randn(batch_size, seq_len, seq_len)
    
    print(f"[INFO] Input shape: {bpp_matrix.shape}")
    
    # Forward pass
    features = processor(bpp_matrix)
    
    print(f"[CHECK] Output shape: {features.shape}")
    assert features.shape == (batch_size, 128), "Output shape mismatch"
    
    print(f"[CHECK] Output dtype: {features.dtype}")
    assert features.dtype == torch.float32, "Output should be float32"
    
    print(f"\n[SUCCESS] BPP processor test passed!")
    return True


def test_feature_fusion():
    """Test feature fusion module."""
    print("\n" + "=" * 80)
    print("TEST 4: Feature Fusion")
    print("=" * 80)
    
    # Create fusion module
    seq_dim = 768
    bpp_dim = 128
    fusion = FeatureFusion(seq_dim=seq_dim, bpp_dim=bpp_dim, output_dim=256)
    
    # Create dummy features
    batch_size = 4
    seq_features = torch.randn(batch_size, seq_dim)
    bpp_features = torch.randn(batch_size, bpp_dim)
    
    print(f"[INFO] Sequence features shape: {seq_features.shape}")
    print(f"[INFO] BPP features shape: {bpp_features.shape}")
    
    # Forward pass
    fused = fusion(seq_features, bpp_features)
    
    print(f"[CHECK] Fused features shape: {fused.shape}")
    assert fused.shape == (batch_size, 256), "Fused shape mismatch"
    
    print(f"\n[SUCCESS] Feature fusion test passed!")
    return True


def test_model_forward():
    """Test complete model forward pass."""
    print("\n" + "=" * 80)
    print("TEST 5: Model Forward Pass")
    print("=" * 80)
    
    # Initialize model
    tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
    model = TEModelWithBPP(
        config_or_model="yangheng/PlantRNA-FM",
        tokenizer=tokenizer,
        num_labels=2,
        label2id={"0": 0, "1": 1}
    )
    
    print(f"[INFO] Model initialized")
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 512
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    bpp_matrix = torch.randn(batch_size, seq_len, seq_len)
    labels = torch.tensor([0, 1])
    
    print(f"[INFO] Input shapes:")
    print(f"  - input_ids: {input_ids.shape}")
    print(f"  - attention_mask: {attention_mask.shape}")
    print(f"  - bpp_matrix: {bpp_matrix.shape}")
    print(f"  - labels: {labels.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bpp_matrix=bpp_matrix,
            labels=labels
        )
    
    print(f"\n[CHECK] Output keys: {list(outputs.keys())}")
    required_keys = ["loss", "logits", "sequence_features", "bpp_features"]
    for key in required_keys:
        assert key in outputs, f"Missing output key: {key}"
    
    print(f"[CHECK] Loss: {outputs['loss'].item():.4f}")
    assert outputs['loss'].item() >= 0, "Loss should be non-negative"
    
    print(f"[CHECK] Logits shape: {outputs['logits'].shape}")
    assert outputs['logits'].shape == (batch_size, 2), "Logits shape mismatch"
    
    print(f"[CHECK] Sequence features shape: {outputs['sequence_features'].shape}")
    print(f"[CHECK] BPP features shape: {outputs['bpp_features'].shape}")
    
    print(f"\n[SUCCESS] Model forward pass test passed!")
    return True


def test_inference():
    """Test model inference."""
    print("\n" + "=" * 80)
    print("TEST 6: Model Inference")
    print("=" * 80)
    
    # Initialize model
    tokenizer = OmniTokenizer.from_pretrained("yangheng/PlantRNA-FM")
    model = TEModelWithBPP(
        config_or_model="yangheng/PlantRNA-FM",
        tokenizer=tokenizer,
        num_labels=2,
        label2id={"0": 0, "1": 1},
        dataset_class=TEDatasetWithBPP
    )
    
    print(f"[INFO] Model initialized for inference")
    
    # Test sequence
    test_sequence = "AUGCAUGCAUGCGCGCGCGC" * 20
    
    print(f"[INFO] Test sequence length: {len(test_sequence)}")
    
    # Run inference
    model.eval()
    outputs = model.inference({"sequence": test_sequence})
    
    print(f"\n[CHECK] Inference output keys: {list(outputs.keys())}")
    
    if "predictions" in outputs:
        prediction = outputs["predictions"][0]
        print(f"[CHECK] Prediction: {prediction} ({'High TE' if prediction == 1 else 'Low TE'})")
        assert prediction in [0, 1], "Prediction should be 0 or 1"
    
    if "confidence" in outputs:
        confidence = outputs["confidence"]
        print(f"[CHECK] Confidence: {confidence:.4f}")
        assert 0 <= confidence <= 1, "Confidence should be in [0, 1]"
    
    print(f"\n[SUCCESS] Inference test passed!")
    return True


def test_training_mini():
    """Test minimal training loop."""
    print("\n" + "=" * 80)
    print("TEST 7: Mini Training Loop")
    print("=" * 80)
    
    from te_bpp_model import train_te_model
    
    print(f"[INFO] Starting mini training with demo data...")
    print(f"[INFO] This will use only 10 samples and 2 epochs for testing")
    
    try:
        results = train_te_model(
            model_name="yangheng/PlantRNA-FM",
            use_demo=True,
            batch_size=2,
            epochs=2,  # Very short training for testing
            output_dir="test_te_model"
        )
        
        print(f"\n[CHECK] Training completed")
        print(f"[CHECK] Results: {results}")
        
        # Validate results
        assert "f1_score" in results or "f1" in results, "Missing F1 score"
        
        print(f"\n[SUCCESS] Training test passed!")
        
        # Cleanup
        import shutil
        if os.path.exists("test_te_model"):
            shutil.rmtree("test_te_model")
        if os.path.exists("translation_efficiency_prediction"):
            shutil.rmtree("translation_efficiency_prediction")
        
        return True
        
    except Exception as e:
        print(f"\n[WARNING] Training test failed: {e}")
        print(f"[INFO] This is expected if GPU/resources are limited")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL TESTS FOR TE-BPP MODEL")
    print("=" * 80)
    
    tests = [
        ("BPP Computation", test_bpp_computation),
        ("Dataset Preparation", test_dataset_preparation),
        ("BPP Processor", test_bpp_processor),
        ("Feature Fusion", test_feature_fusion),
        ("Model Forward", test_model_forward),
        ("Model Inference", test_inference),
        ("Mini Training", test_training_mini),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception:")
            print(f"  {str(e)}")
            results[test_name] = "ERROR"
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "ERROR": "⚠"
        }.get(result, "?")
        print(f"  {status_symbol} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASS")
    total = len(results)
    
    print(f"\n[RESULT] {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! 🎉")
        return True
    else:
        print("[WARNING] Some tests failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
