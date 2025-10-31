# -*- coding: utf-8 -*-
# file: conftest.py
# time: 18:05 31/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Pytest configuration and shared fixtures.
"""

import pytest
import warnings


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "cpu: marks tests that run on CPU only"
    )


@pytest.fixture(scope="session")
def test_model_name():
    """Default model name for testing"""
    return "yangheng/OmniGenome-186M"


@pytest.fixture(scope="session")
def test_sequences():
    """Common test sequences for genomic tasks"""
    return [
        "ATCGATCGATCGATCGATCGATCGATCGATCG",
        "GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        "TATATATATATATATATATATATATATATATAT",
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAG",
        "CTCTCTCTCTCTCTCTCTCTCTCTCTCTCTCT"
    ]


@pytest.fixture(scope="session")
def test_rna_sequences():
    """Common RNA sequences for testing"""
    return [
        "GGCCUUAGCUCAGCGGUUAGAGCGCACCCCUGAUAAGGGUGAGGUCGCUGAUUCGAUCCAGCUAAGGCCACCA",
        "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA",
        "AGAGUGGUUGACUCGUGUGCGCGCGAGCGAGUAGCAAAGCGAGGUCGCUGGUUCGAUUCCGGCACCUCUCU"
    ]


@pytest.fixture(autouse=True)
def setup_warnings():
    """Configure warnings for tests"""
    # Ignore deprecation warnings from dependencies
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    
    yield
    
    # Reset warnings after test
    warnings.resetwarnings()


@pytest.fixture
def temp_test_data(tmp_path):
    """Create temporary test data directory"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Automatically mark tests based on their names
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "pipeline" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if "unit" in item.nodeid.lower() or item.parent.name.startswith("Test"):
            item.add_marker(pytest.mark.unit)
