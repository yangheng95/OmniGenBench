# -*- coding: utf-8 -*-
# file: test_hf_download.py
# time: 15:45 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Tests for robust HuggingFace Hub downloading without git-lfs.
"""

import os
import shutil
import pytest
from pathlib import Path

try:
    from omnigenbench.src.utility.model_hub.hf_download import (
        download_from_hf_hub,
        verify_download_integrity,
        list_hf_repo_files,
        get_model_info,
        HF_HUB_AVAILABLE,
    )
    SKIP_TESTS = not HF_HUB_AVAILABLE
except ImportError:
    SKIP_TESTS = True

from omnigenbench import ModelHub


@pytest.mark.skipif(SKIP_TESTS, reason="huggingface_hub not available")
class TestHFDownload:
    """Test HuggingFace Hub downloading functionality."""

    @pytest.fixture
    def test_cache_dir(self, tmp_path):
        """Create a temporary cache directory for testing."""
        cache_dir = tmp_path / "test_cache"
        cache_dir.mkdir()
        yield str(cache_dir)
        # Cleanup
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    def test_download_small_model(self, test_cache_dir):
        """Test downloading a small model using HF Hub API."""
        # Use a small model for testing
        model_id = "hf-internal-testing/tiny-random-bert"
        
        path = download_from_hf_hub(
            repo_id=model_id,
            cache_dir=test_cache_dir,
            force_download=False,
        )
        
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "config.json"))
        
    def test_download_with_force(self, test_cache_dir):
        """Test force re-download functionality."""
        model_id = "hf-internal-testing/tiny-random-bert"
        
        # First download
        path1 = download_from_hf_hub(
            repo_id=model_id,
            cache_dir=test_cache_dir,
            force_download=False,
        )
        
        # Second download with force
        path2 = download_from_hf_hub(
            repo_id=model_id,
            cache_dir=test_cache_dir,
            force_download=True,
        )
        
        assert path1 == path2
        assert os.path.exists(path2)

    def test_download_with_patterns(self, test_cache_dir):
        """Test downloading only specific file patterns."""
        model_id = "hf-internal-testing/tiny-random-bert"
        
        path = download_from_hf_hub(
            repo_id=model_id,
            cache_dir=test_cache_dir,
            allow_patterns=["*.json"],  # Only JSON files
        )
        
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "config.json"))

    def test_verify_download_integrity(self, test_cache_dir):
        """Test download integrity verification."""
        model_id = "hf-internal-testing/tiny-random-bert"
        
        path = download_from_hf_hub(
            repo_id=model_id,
            cache_dir=test_cache_dir,
        )
        
        # Should pass with default checks
        assert verify_download_integrity(path)
        
        # Should pass with specific file checks
        assert verify_download_integrity(
            path, 
            required_files=["config.json", "pytorch_model.bin"]
        )

    def test_list_repo_files(self):
        """Test listing files in a repository."""
        model_id = "hf-internal-testing/tiny-random-bert"
        
        files = list_hf_repo_files(model_id)
        
        assert isinstance(files, list)
        assert len(files) > 0
        assert "config.json" in files

    def test_get_model_info(self):
        """Test getting model metadata."""
        model_id = "hf-internal-testing/tiny-random-bert"
        
        info = get_model_info(model_id)
        
        assert isinstance(info, dict)
        assert "id" in info
        assert "siblings" in info


@pytest.mark.slow
class TestModelHubWithHFAPI:
    """Test ModelHub integration with new HF Hub API downloader."""

    @pytest.fixture
    def test_model_name(self):
        """Small test model for integration testing."""
        return "hf-internal-testing/tiny-random-bert"

    def test_modelhub_load_with_hf_api(self, test_model_name):
        """Test ModelHub.load() using HF Hub API."""
        # This should use the new download_hf_model with use_hf_api=True
        model = ModelHub.load(
            test_model_name,
            device="cpu",
            use_hf_api=True,  # Force use of HF API
        )
        
        assert model is not None
        assert hasattr(model, "tokenizer")

    def test_modelhub_load_fallback_to_git(self, test_model_name):
        """Test ModelHub.load() fallback to git clone."""
        # This tests the fallback mechanism
        model = ModelHub.load(
            test_model_name,
            device="cpu",
            use_hf_api=False,  # Force use of git clone
        )
        
        assert model is not None
        assert hasattr(model, "tokenizer")


@pytest.mark.skipif(SKIP_TESTS, reason="huggingface_hub not available")
def test_detect_lfs_pointer_file(tmp_path):
    """Test detection of git-lfs pointer files."""
    # Create a fake LFS pointer file
    fake_model_dir = tmp_path / "fake_model"
    fake_model_dir.mkdir()
    
    # Create config.json
    config_path = fake_model_dir / "config.json"
    config_path.write_text('{"model_type": "bert"}')
    
    # Create a fake LFS pointer file
    pointer_path = fake_model_dir / "pytorch_model.bin"
    pointer_content = """version https://git-lfs.github.com/spec/v1
oid sha256:b437d27531abc123
size 41943280
"""
    pointer_path.write_text(pointer_content)
    
    # Verify should detect the pointer file
    result = verify_download_integrity(str(fake_model_dir))
    assert result is False  # Should fail due to LFS pointer


def test_model_loading_comparison():
    """
    Compare model loading with different methods.
    
    This test documents the differences between git-lfs and HF Hub API.
    """
    print("\n" + "="*80)
    print("Model Loading Method Comparison")
    print("="*80)
    
    comparison = """
    Method 1: Git Clone (Original)
    ├── Requires: git + git-lfs
    ├── Download: Via git protocol
    ├── Files: Full git history included
    ├── Size: Larger (includes .git directory)
    ├── Speed: Can be slower for large models
    └── Risk: LFS pointer files if git-lfs not installed
    
    Method 2: HF Hub API (New, Recommended)
    ├── Requires: huggingface_hub package only
    ├── Download: Direct HTTPS from CDN
    ├── Files: Model files only (no git history)
    ├── Size: Smaller (no .git directory)
    ├── Speed: Faster, chunked downloads
    └── Risk: None - always gets actual files
    
    Usage:
    ------
    # Automatic (uses HF API by default)
    model = ModelHub.load("yangheng/OmniGenome-186M")
    
    # Force HF API
    model = ModelHub.load("yangheng/OmniGenome-186M", use_hf_api=True)
    
    # Force git clone (legacy)
    model = ModelHub.load("yangheng/OmniGenome-186M", use_hf_api=False)
    """
    
    print(comparison)
    print("="*80)


if __name__ == "__main__":
    # Run comparison documentation
    test_model_loading_comparison()
    
    # Run tests
    pytest.main([__file__, "-v", "-s"])
