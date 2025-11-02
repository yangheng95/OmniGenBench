# OmniGenBench AI Coding Agent Instructions

## Project Overview

OmniGenBench is a unified framework for **genomic foundation model (GFM)** development, benchmarking, and deployment. It provides automated workflows for training, inference, and evaluation of 30+ pre-trained models on RNA/DNA sequences across 80+ benchmark tasks.

**Key Design Philosophy**: Plug-and-play modularity through abstract base classes (`OmniModel`, `OmniDataset`, `OmniMetric`, `OmniTokenizer`) that enable seamless integration of custom components.

## Architecture at a Glance

```
omnigenbench/
├── auto/                    # Automated workflows (AutoBench, AutoTrain)
│   ├── auto_bench/         # Benchmark orchestration
│   └── auto_train/         # Training orchestration
├── cli/                    # Command-line interfaces
│   ├── ogb_cli.py          # Unified CLI entry point (ogb command)
│   ├── omnigenome_cli.py   # Specialized CLI (RNA design, etc.)
│   └── commands/           # Command implementations
├── src/                    # Core library components
│   ├── abc/                # Abstract base classes (CRITICAL - read these first)
│   ├── model/              # Model implementations by task type
│   ├── dataset/            # Dataset loaders
│   ├── metric/             # Evaluation metrics
│   └── tokenizer/          # Sequence tokenization
└── tests/                  # Pattern-based tests
```

## Critical Development Patterns

### 1. Abstract Base Classes (The Foundation)
**ALWAYS read these before implementing new components:**
- `src/abc/abstract_model.py` - All models inherit from `OmniModel`
- `src/abc/abstract_dataset.py` - Dataset interface contract
- `src/abc/abstract_tokenizer.py` - Tokenization patterns
- `src/abc/abstract_metric.py` - Evaluation interface

**Key Pattern**: Models auto-load from HuggingFace via `config.auto_map` or `architectures` field:
```python
# OmniModel automatically resolves the correct architecture
model = OmniModelForSequenceClassification(
    model="yangheng/OmniGenome-186M",
    num_labels=2
)
```

### 2. Task-Specific Model Classes
Models are organized by task type, NOT by architecture:
- `OmniModelForSequenceClassification` - DNA/RNA sequence-level classification
- `OmniModelForTokenClassification` - Per-nucleotide predictions (e.g., splice sites)
- `OmniModelForRNADesign` - Generative tasks (genetic algorithms + MLM)
- `OmniModelForEmbedding` - Feature extraction

**Pattern**: Task type determines the forward pass and loss function, NOT the underlying transformer.

### 3. CLI Command Structure
Two entry points serve different purposes:
- **`ogb`** (unified CLI): For core workflows (autoinfer, autotrain, autobench, rna_design)
- **`omnigenome_cli`**: Legacy specialized commands

**Adding Commands to `ogb`**:
1. Create parser function: `create_<command>_parser(subparsers)`
2. Create handler: `run_<command>(args)`
3. Register in `main()`: `create_<command>_parser(subparsers)`
4. Handler calls the actual implementation in `cli/commands/<domain>/<command>.py`

**Command Implementation Pattern** (see `cli/commands/base.py`):
```python
class MyCommand(BaseCommand):
    @classmethod
    def register_command(cls, subparsers):
        parser = subparsers.add_parser("mycommand")
        parser.add_argument("--arg", required=True)
        cls.add_common_arguments(parser)  # Adds --log-level, --output-dir
        parser.set_defaults(func=cls.execute)
    
    @staticmethod
    def execute(args: argparse.Namespace):
        # Implementation using args directly
        pass
```

### 4. Trainer Selection System
Three trainer types handle different execution contexts:
- **`native`**: Pure PyTorch training loop
- **`accelerate`**: Distributed training via HuggingFace Accelerate
- **`hf_trainer`**: HuggingFace Trainer API integration

**Pattern**: Select via `--trainer` flag. `accelerate` is default for benchmarking.

### 5. Dataset Configuration Pattern
Datasets use JSON files + config dicts (NOT HuggingFace datasets API directly):
```python
# Config pattern from examples/autobench_gfm_evaluation/RGB/*/config.py
config = {
    "task_type": "sequence_classification",  # Determines dataset class
    "num_labels": 2,
    "train_file": "path/to/train.json",
    "test_file": "path/to/test.json",
    "max_length": 512,
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 2e-5,
    "seeds": [0, 1, 2],  # Multi-seed evaluation
    "compute_metrics": [ClassificationMetric().accuracy]
}
```

### 6. RNA-Specific Workflows
RNA design uses **genetic algorithms enhanced with MLM**:
- ViennaRNA for structure prediction (`viennarna` package)
- Multi-objective optimization (structure similarity + free energy)
- Progress tracking with `tqdm` and early termination
- Returns list of sequences (up to 25), never single string

**Key Files**:
- `src/model/rna_design/model.py` - Core algorithm
- `cli/commands/rna/rna_design.py` - CLI implementation

## Development Workflows

### Running Tests
```bash
# All tests
pytest

# Specific patterns (tests are pattern-based, not integration)
pytest tests/test_training_patterns.py
pytest -m "not slow"  # Skip slow tests
pytest --cov=omnigenbench --cov-report=html  # With coverage
```

### Installing Development Mode
```bash
pip install -e .  # Editable install for development
```

### Benchmark Execution
```bash
# CLI approach
ogb autobench --model yangheng/OmniGenome-186M --benchmark RGB --trainer accelerate

# Python API approach (see examples/autobench_gfm_evaluation/)
from omnigenbench import AutoBench
bench = AutoBench(benchmark="RGB", config_or_model="model", overwrite=False)
bench.run(batch_size=8, seeds=[0, 1, 2])
```

## Project-Specific Conventions

### 1. Import Pattern
```python
# User-facing imports from top-level
from omnigenbench import (
    OmniModelForSequenceClassification,
    OmniDatasetForSequenceClassification,
    AutoBench, AutoTrain, ModelHub
)
```

### 2. File Headers
All files include standardized headers with author info:
```python
# -*- coding: utf-8 -*-
# file: filename.py
# time: HH:MM DD/MM/YYYY
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.
```

### 3. Docstring Style
Detailed docstrings with examples for all public methods:
```python
def method(self, arg):
    """
    One-line summary.
    
    Detailed description explaining purpose and algorithm.
    
    Args:
        arg (type): Description
        
    Returns:
        type: Description
        
    Example:
        >>> result = method("value")
        >>> print(result)
    """
```

### 4. Windows Compatibility
- **AVOID** emoji in console output (causes encoding errors)
- Use text markers: `[SUCCESS]`, `[INFO]`, `[ERROR]` instead
- Terminal commands use bash (via Git Bash or WSL)

### 5. Configuration Management
- `AutoConfig` centralizes all configuration
- Configs use plain dicts, not complex nested objects
- Validation happens at runtime in trainer/model initialization

### 6. Model Loading from HuggingFace Hub
- Models auto-detect from `config.json` via `auto_map` or `architectures`
- Trust remote code handling: `trust_remote_code=True` for custom models
- Tokenizers loaded separately but matched to model by convention

## Integration Points

### HuggingFace Hub
- Models: `ModelHub.load()` wraps HF Hub with OmniModel interface
- Datasets: Direct download via HF datasets library
- Fine-tuned models stored at: `yangheng/ogb_*_finetuned`

### ViennaRNA (RNA Structure Prediction)
- Used in RNA design and structure prediction tasks
- Python bindings: `import RNA` from `viennarna` package
- Free energy calculation and secondary structure folding

### AutoCUDA (Device Management)
```python
import autocuda
device = autocuda.auto_cuda()  # Automatic GPU detection
```

## Common Pitfalls & Solutions

### ❌ Don't: Create models without task type
```python
model = OmniModel("yangheng/OmniGenome-186M")  # Too abstract!
```
✅ Do: Use task-specific model class
```python
model = OmniModelForSequenceClassification("yangheng/OmniGenome-186M", num_labels=2)
```

### ❌ Don't: Import from internal paths
```python
from omnigenbench.src.model.sequence_classification import Model
```
✅ Do: Import from top-level API
```python
from omnigenbench import OmniModelForSequenceClassification
```

### ❌ Don't: Assume single return value for RNA design
```python
sequence = model.design(structure="(((...)))")  # May be list!
```
✅ Do: Always handle as list
```python
sequences = model.design(structure="(((...)))")
for seq in sequences:
    print(seq)
```

### ❌ Don't: Use platform-specific paths
```python
output_dir = "results\\model"  # Windows-specific
```
✅ Do: Use forward slashes or Path objects
```python
from pathlib import Path
output_dir = Path("results/model")
```

## Quick Reference

### Supported Benchmarks
- **RGB**: RNA structure + function (12 tasks)
- **BEACON**: Multi-domain RNA (13 tasks)
- **PGB**: Plant genomics (7 categories)
- **GUE**: DNA general understanding (36 datasets)
- **GB**: Classic DNA classification (9 datasets)

### Key Configuration Files
- `setup.py` - Package metadata and entry points
- `pytest.ini` - Test configuration (coverage target: 80%)
- `requirements.txt` - Core dependencies
- `examples/*/config.py` - Task-specific training configs

### Documentation Structure
- `docs/GETTING_STARTED.md` - Comprehensive user guide
- `docs/cli.rst` - CLI reference documentation
- `examples/` - Working code examples (preferred over docs for patterns)
- `framework_architecture_v2.md` - Architecture diagrams (in Chinese)

## When You Need Help

1. **Understanding a component**: Read the abstract base class in `src/abc/` first
2. **Adding a new task type**: Check existing model classes in `src/model/`
3. **CLI patterns**: Look at `cli/commands/rna/rna_design.py` as reference
4. **Training patterns**: See `tests/test_training_patterns.py` for canonical examples
5. **Dataset formats**: Check `examples/autobench_gfm_evaluation/*/train.json`

---

**Version**: 0.3.23alpha  
**Last Updated**: January 2025  
**Maintained By**: YANG, HENG (yangheng95)
