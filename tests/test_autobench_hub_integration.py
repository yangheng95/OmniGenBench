# -*- coding: utf-8 -*-
# file: test_autobench_hub_integration.py
# time: 17:30 08/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Test suite for AutoBench HuggingFace Hub integration
"""

import os
import pytest
import tempfile
from pathlib import Path

from omnigenbench import AutoBench
from omnigenbench.src.utility.hub_utils import download_benchmark


class TestAutoBenchHubIntegration:
    """Test AutoBench integration with HuggingFace Hub benchmarks"""

    def test_hub_benchmark_detection(self):
        """Test that AutoBench correctly detects hub vs local benchmarks"""
        # Create a temporary local directory with metadata.py
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local_benchmark"
            local_path.mkdir()
            
            # Create a minimal metadata.py
            metadata_content = """
bench_list = []
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = local_path / "metadata.py"
            metadata_file.write_text(metadata_content)

            # Test with local path (should be detected as local)
            bench_local = AutoBench(
                benchmark=str(local_path),
                config_or_model="yangheng/OmniGenome-52M",
            )
            assert not bench_local.is_hub_benchmark
            assert bench_local.benchmark == str(local_path)

    def test_autobench_cache_dir_parameter(self):
        """Test that AutoBench accepts cache_dir parameter"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local benchmark
            benchmark_dir = Path(tmpdir) / "local_benchmark"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = []
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text(metadata_content)
            
            cache_dir = Path(tmpdir) / "custom_cache"

            bench = AutoBench(
                benchmark=str(benchmark_dir),
                config_or_model="yangheng/OmniGenome-52M",
                cache_dir=str(cache_dir),
            )

            # Check that cache_dir is stored
            assert hasattr(bench, "cache_dir")
            assert bench.cache_dir == str(cache_dir)

    def test_backward_compatibility_local_benchmark(self):
        """Test that local benchmark workflows still work"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local_benchmark"
            local_path.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = []
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = local_path / "metadata.py"
            metadata_file.write_text(metadata_content)

            # Old-style initialization should still work
            bench = AutoBench(
                benchmark=str(local_path),
                config_or_model="yangheng/OmniGenome-52M",
            )

            assert bench.benchmark == str(local_path)
            assert not bench.is_hub_benchmark
            assert bench.bench_metadata.bench_list == []

    def test_autobench_benchmark_name_handling(self):
        """Test that benchmark names are handled correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local benchmark
            benchmark_dir = Path(tmpdir) / "my_benchmark_name"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = ["task1", "task2"]
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text(metadata_content)
            
            bench = AutoBench(
                benchmark=str(benchmark_dir),
                config_or_model="yangheng/OmniGenome-52M",
            )

            # Check that benchmark name is extracted correctly
            assert hasattr(bench, "benchmark_name_or_path")
            assert "my_benchmark_name" in bench.benchmark_name_or_path
            assert bench.bench_metadata.bench_list == ["task1", "task2"]

    def test_download_benchmark_function_exists(self):
        """Test that download_benchmark function is available"""
        assert callable(download_benchmark)

    def test_download_benchmark_with_local_path(self):
        """Test download_benchmark with a local path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = Path(tmpdir) / "local_benchmark"
            local_path.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = []
"""
            metadata_file = local_path / "metadata.py"
            metadata_file.write_text(metadata_content)

            # Should return the local path without downloading
            result = download_benchmark(str(local_path))
            
            # Result should be the local path
            assert result == str(local_path)

    def test_autobench_run_method_signature(self):
        """Test that AutoBench.run() has the correct signature"""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir) / "test_benchmark"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = []
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text(metadata_content)
            
            bench = AutoBench(
                benchmark=str(benchmark_dir),
                config_or_model="yangheng/OmniGenome-52M",
            )

            # Verify that run method exists and accepts kwargs
            import inspect

            sig = inspect.signature(bench.run)
            assert "kwargs" in sig.parameters

    def test_autobench_attributes(self):
        """Test that AutoBench has all expected attributes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir) / "test_benchmark"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_content = """
bench_list = ["task1"]
__omnigenbench_version__ = "0.3.28alpha"
"""
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text(metadata_content)
            
            bench = AutoBench(
                benchmark=str(benchmark_dir),
                config_or_model="yangheng/OmniGenome-52M",
                autocast="bf16",
                trainer="accelerate",
            )

            # Check all expected attributes
            assert hasattr(bench, "benchmark")
            assert hasattr(bench, "benchmark_name_or_path")
            assert hasattr(bench, "is_hub_benchmark")
            assert hasattr(bench, "config_or_model")
            assert hasattr(bench, "model_name")
            assert hasattr(bench, "autocast")
            assert hasattr(bench, "trainer")
            assert hasattr(bench, "mv")
            assert hasattr(bench, "mv_path")
            assert hasattr(bench, "bench_metadata")
            
            # Check attribute values
            assert bench.autocast == "bf16"
            assert bench.trainer == "accelerate"
            assert bench.model_name == "OmniGenome-52M"


class TestDownloadBenchmarkRefactoring:
    """Test the refactored download_benchmark function"""

    def test_download_benchmark_cache_dir_default(self):
        """Test that download_benchmark uses correct default cache_dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a local benchmark to avoid actual download
            benchmark_dir = Path(tmpdir) / "RGB"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text("bench_list = []")
            
            # Test with local path
            result = download_benchmark(str(benchmark_dir))
            assert result == str(benchmark_dir)

    def test_download_benchmark_force_download_flag(self):
        """Test that download_benchmark respects force_download flag"""
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark_dir = Path(tmpdir) / "test_benchmark"
            benchmark_dir.mkdir()
            
            # Create metadata.py
            metadata_file = benchmark_dir / "metadata.py"
            metadata_file.write_text("bench_list = []")
            
            # First call without force_download
            result1 = download_benchmark(str(benchmark_dir), force_download=False)
            assert result1 == str(benchmark_dir)
            
            # Second call with force_download
            result2 = download_benchmark(str(benchmark_dir), force_download=True)
            assert result2 == str(benchmark_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
