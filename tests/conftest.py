"""
Pytest configuration and shared fixtures for OmniGenBench tests.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark slow tests and skip GPU tests if CUDA not available."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.nodeid or "model_loading" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Skip GPU tests if CUDA not available
        if item.get_closest_marker("gpu") and not cuda_available:
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))


@pytest.fixture
def sample_rna_sequences():
    """Sample RNA sequences for testing."""
    return [
        "AUGGCUACG",
        "CGGAUACGGC", 
        "UGGCCAAGUC",
        "AUGCUGCUAUGCUA"
    ]


@pytest.fixture
def sample_rna_structures():
    """Sample RNA secondary structures for testing."""
    return [
        "(((())))",
        "(((...)))",
        "........",
        "((..))"
    ]


@pytest.fixture
def sample_dataset_entries():
    """Sample dataset entries in the format used by examples."""
    return [
        {"seq": "AUCG", "label": "(..)"},
        {"seq": "AUGC", "label": "().."},
        {"seq": "CGAU", "label": "(())"},
        {"seq": "GAUC", "label": "...."}
    ]


@pytest.fixture
def mock_model_config():
    """Mock model configuration for testing."""
    from unittest.mock import MagicMock
    config = MagicMock()
    config.hidden_size = 768
    config.num_labels = 2
    config.vocab_size = 32
    config.max_position_embeddings = 512
    return config


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    from unittest.mock import MagicMock
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "AUGC"
    tokenizer.convert_ids_to_tokens.return_value = ["A", "U", "G", "C"]
    tokenizer.vocab_size = 32
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    return tokenizer


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with sample data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample train.json
    train_file = data_dir / "train.json"
    train_data = [
        '{"seq": "AUCG", "label": "(..)"}',
        '{"seq": "AUGC", "label": "().."}',
        '{"seq": "CGAU", "label": "(())"}'
    ]
    train_file.write_text("\n".join(train_data))
    
    # Create sample test.json
    test_file = data_dir / "test.json"
    test_data = [
        '{"seq": "GAUC", "label": "...."}',
        '{"seq": "UCGA", "label": "(.)"}'
    ]
    test_file.write_text("\n".join(test_data))
    
    # Create sample config.py
    config_file = data_dir / "config.py"
    config_content = '''
# Dataset configuration
max_length = 512
num_labels = 4
task_type = "classification"
'''
    config_file.write_text(config_content)
    
    return data_dir


@pytest.fixture(scope="session")
def examples_dir():
    """Path to examples directory."""
    return ROOT_DIR / "examples"


@pytest.fixture
def skip_if_no_omnigenome():
    """Skip test if omnigenbench package is not available."""
    try:
        import omnigenbench
        return False
    except ImportError:
        pytest.skip("omnigenome package not available")


# Custom pytest markers
pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:.*:UserWarning"),
] 