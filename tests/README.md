# OmniGenBench Test Suite

Comprehensive test suite for OmniGenBench functionality, based entirely on the examples provided in the `examples/` directory.

## Overview

This test suite provides thorough coverage of OmniGenBench's core functionalities:

- **RNA Sequence Design** - Genetic algorithm-based RNA design
- **Attention Score Extraction** - Multi-layer attention analysis
- **AutoInfer CLI** - Model inference from various input formats
- **Structure Prediction** - RNA secondary structure prediction
- **Genomic Embeddings** - Sequence embedding extraction and similarity
- **Token Classification/Regression** - Per-nucleotide predictions

## Test Files

### `test_rna_design.py`
Based on: `examples/rna_sequence_design/rna_design_examples.py`

Tests RNA sequence design using genetic algorithms with MLM-guided mutations:
- Simple hairpin structures
- Custom evolutionary parameters
- Batch design of multiple structures
- Sequence validation with ViennaRNA
- Edge cases (very short, no base pairs, complex nested)
- Performance tests (large populations, longer structures)

**Key Test Classes:**
- `TestRNADesignBasic` - Basic design functionality
- `TestRNADesignBatch` - Batch processing
- `TestRNADesignValidation` - Sequence validation
- `TestRNADesignEdgeCases` - Edge case handling
- `TestRNADesignPerformance` - Performance tests (slow)

### `test_attention_extraction.py`
Based on: `examples/attention_score_extraction/attention_extraction_example.py`

Tests attention score extraction across all model types:
- Single sequence attention extraction
- Layer and head selection
- Attention statistics (entropy, concentration)
- Batch attention extraction
- Cross-model compatibility (Embedding, Classification, Regression)
- Edge cases (short sequences, max length, invalid indices)

**Key Test Classes:**
- `TestAttentionExtractionEmbeddingModel` - Embedding model tests
- `TestAttentionExtractionBatch` - Batch processing
- `TestAttentionExtractionTaskModels` - Task model compatibility
- `TestAttentionExtractionEdgeCases` - Edge cases
- `TestAttentionExtractionPerformance` - Performance tests (slow)

### `test_autoinfer_cli.py`
Based on: `examples/autoinfer_examples/README.md`

Tests model inference capabilities and input/output formats:
- Single sequence inference
- Batch inference
- JSON file formats (simple list, with metadata)
- CSV file format
- Text file format
- Output format validation
- Full pipeline integration

**Key Test Classes:**
- `TestModelHubInference` - Core inference functionality
- `TestInputFileFormats` - File format handling
- `TestOutputFormat` - Output structure validation
- `TestInferenceEdgeCases` - Edge cases
- `TestBatchProcessing` - Batch processing
- `TestRealWorldScenarios` - Integration tests

### `test_structure_prediction.py`
Based on: `examples/rna_secondary_structure_prediction/`

Tests RNA secondary structure prediction:
- Structure validation (bracket matching)
- Base pair extraction
- Structure comparison metrics (accuracy, precision, recall, F1)
- Model inference
- ViennaRNA comparison
- Structure fixing for invalid brackets

**Key Test Classes:**
- `TestStructureValidation` - Structure validation functions
- `TestBasePairExtraction` - Base pair extraction
- `TestStructureMetrics` - Comparison metrics
- `TestModelInference` - Model prediction
- `TestViennaRNAComparison` - ViennaRNA integration
- `TestEdgeCases` - Edge cases
- `TestStructurePredictionPerformance` - Performance tests (slow)

### `test_genomic_embeddings.py`
Based on: `examples/genomic_embeddings/README.md`

Tests genomic sequence embedding extraction:
- Single sequence encoding
- Aggregation strategies (mean, head, tail)
- Batch encoding
- Sequence similarity computation
- Embedding properties (normalization, dimensionality, variance)
- Edge cases (short/long sequences, empty batches)

**Key Test Classes:**
- `TestSingleSequenceEmbedding` - Single sequence encoding
- `TestAggregationStrategies` - Different aggregation methods
- `TestBatchEmbedding` - Batch processing
- `TestSequenceSimilarity` - Similarity computation
- `TestEmbeddingProperties` - Mathematical properties
- `TestEdgeCases` - Edge cases
- `TestEmbeddingPerformance` - Performance tests (slow)

### `test_token_classification.py`
Based on: `examples/autobench_gfm_evaluation/RGB/*/config.py`

Tests token-level classification and regression:
- Token classification configuration (structure prediction)
- Token regression configuration (mRNA degradation)
- Model initialization and forward pass
- Classification and regression metrics
- Loss functions with padding
- Dataset structure validation
- Complete pipelines

**Key Test Classes:**
- `TestTokenClassificationConfig` - Configuration tests
- `TestTokenClassificationModel` - Classification model tests
- `TestTokenRegressionConfig` - Regression configuration
- `TestTokenRegressionModel` - Regression model tests
- `TestMetrics` - Metric computation
- `TestDatasetPreparation` - Dataset structure
- `TestLossFunction` - Loss computation
- `TestTokenLevelPipeline` - Integration tests

### `conftest.py`
Shared pytest configuration and fixtures:
- Custom markers (slow, integration, unit, gpu, cpu)
- Shared fixtures (model names, test sequences)
- Warning configuration
- Automatic test marking

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_rna_design.py
```

### Run Specific Test Class
```bash
pytest tests/test_rna_design.py::TestRNADesignBasic
```

### Run Specific Test
```bash
pytest tests/test_rna_design.py::TestRNADesignBasic::test_simple_hairpin_design
```

### Skip Slow Tests
```bash
pytest tests/ -m "not slow"
```

### Run Only Integration Tests
```bash
pytest tests/ -m integration
```

### Run with Coverage
```bash
pytest tests/ --cov=omnigenbench --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Tests by Keyword
```bash
pytest tests/ -k "batch"  # Run all tests with "batch" in name
pytest tests/ -k "embedding"  # Run all embedding-related tests
```

## Test Markers

Tests are marked with the following markers:

- **`@pytest.mark.slow`** - Tests that take longer to run (e.g., large populations, many generations)
- **`@pytest.mark.integration`** - End-to-end integration tests
- **`@pytest.mark.unit`** - Unit tests (automatically applied)
- **`@pytest.mark.gpu`** - Tests requiring GPU (not currently used)
- **`@pytest.mark.cpu`** - CPU-only tests (not currently used)

## Test Structure

Each test file follows a consistent pattern:

```python
import pytest
from omnigenbench import ...

@pytest.fixture(scope="module")
def shared_resource():
    """Load once per module"""
    return ...

class TestFeatureBasic:
    """Basic functionality tests"""
    
    def test_something(self):
        """Test description"""
        # Arrange
        # Act
        # Assert

class TestFeatureEdgeCases:
    """Edge case tests"""
    # ...

@pytest.mark.slow
class TestFeaturePerformance:
    """Performance tests (marked as slow)"""
    # ...
```

## Dependencies

Tests depend on the core OmniGenBench package:

```python
from omnigenbench import (
    OmniModelForRNADesign,
    OmniModelForEmbedding,
    OmniModelForSequenceClassification,
    OmniModelForTokenClassification,
    OmniTokenizer,
    ModelHub,
    AutoConfig,
    # ...
)
```

Optional dependencies for specific tests:
- **ViennaRNA** - Required for RNA structure validation tests
- **matplotlib** - For visualization tests (if added)

## Design Principles

### 1. Based on Real Examples
All tests are derived from actual code in the `examples/` directory. No hallucinated functionality.

### 2. Comprehensive Coverage
Each test file covers:
- Basic functionality
- Edge cases
- Error handling
- Integration scenarios
- Performance (marked as slow)

### 3. Independent Tests
Tests are independent and can run in any order. Shared resources use fixtures with appropriate scopes.

### 4. Clear Documentation
Each test includes:
- Docstring explaining what is tested
- Reference to source example
- Clear arrange-act-assert structure

### 5. Realistic Test Data
Test data matches the format and structure used in examples:
- Real RNA/DNA sequences
- Actual structure notations
- Valid label formats

## Common Patterns

### Model Loading
```python
@pytest.fixture(scope="module")
def model(model_name):
    """Load model once for all tests"""
    return OmniModelForSomething(model=model_name)
```

### File I/O Testing
```python
def test_save_and_load(tmp_path):
    """Use tmp_path fixture for temporary files"""
    output_file = tmp_path / "results.json"
    # ... save to output_file
    assert output_file.exists()
```

### Validation Patterns
```python
assert isinstance(result, expected_type), "Type check"
assert len(result) > 0, "Non-empty check"
assert all(condition for item in result), "Element-wise check"
```

## Troubleshooting

### Test Failures

**Model loading errors:**
```bash
# Ensure you have internet connection to download models
# Or use cached models in __OMNIGENBENCH_DATA__/
```

**ViennaRNA not found:**
```bash
# Install ViennaRNA
pip install ViennaRNA
# Or skip those tests
pytest -k "not vienna"
```

**Slow tests timing out:**
```bash
# Skip slow tests
pytest -m "not slow"
# Or increase timeout
pytest --timeout=300
```

### Running Subset of Tests

```bash
# Only RNA-related tests
pytest tests/ -k "rna"

# Only embedding tests
pytest tests/ -k "embedding"

# Fast tests only
pytest tests/ -m "not slow"
```

## Contributing

When adding new tests:

1. **Base on examples** - Only test functionality present in `examples/`
2. **Follow naming** - Use descriptive test names: `test_<functionality>_<scenario>`
3. **Add docstrings** - Reference the source example file
4. **Use fixtures** - Share resources via fixtures
5. **Mark appropriately** - Add `@pytest.mark.slow` for slow tests
6. **Document** - Update this README with new test coverage

## Coverage Goals

Current coverage (based on examples):
- ✅ RNA sequence design
- ✅ Attention extraction
- ✅ AutoInfer CLI
- ✅ Structure prediction
- ✅ Genomic embeddings
- ✅ Token classification/regression

Future coverage (if examples added):
- ⬜ Variant effect prediction
- ⬜ Transcription factor binding
- ⬜ Translation efficiency
- ⬜ Data augmentation pipelines
- ⬜ Custom model training

## References

All tests are based on code in:
- `examples/rna_sequence_design/`
- `examples/attention_score_extraction/`
- `examples/autoinfer_examples/`
- `examples/rna_secondary_structure_prediction/`
- `examples/genomic_embeddings/`
- `examples/genomic_data_augmentation/`
- `examples/autobench_gfm_evaluation/RGB/*/config.py`

See individual test files for specific example references.
