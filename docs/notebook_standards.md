# Notebook Design Standards for OmniGenBench Tutorials

**Version**: 1.0  
**Last Updated**: November 2024  
**Author**: YANG, HENG  

## Philosophy: Reproducibility-First Design

All OmniGenBench tutorial notebooks follow a unified design philosophy prioritizing:

1. **Reproducibility**: One-click "Restart & Run All" must succeed every time
2. **Determinism by Default**: All randomness controlled via explicit seeds
3. **Separation of Concerns**: Clear boundaries between config, data, compute, and visualization
4. **Progressive Disclosure**: Simple defaults with advanced options clearly documented
5. **POLA (Principle of Least Astonishment)**: Consistent patterns across all tutorials
6. **Single Source of Truth (SSoT)**: Configuration centralized, no duplicated values

---

## Unified Structure

### Standard Notebook Anatomy

Every tutorial notebook follows this canonical structure:

```
📓 Tutorial Notebook
│
├── 📖 Header Cell (Markdown)
│   ├── Title with emoji
│   ├── Learning objectives
│   ├── Biological/computational context
│   └── 4-step workflow diagram (Mermaid)
│
├── 🚀 Step 1: Setup and Configuration
│   ├── 1.1: Environment Setup (one cell)
│   │   └── pip install + shared_utils import
│   ├── 1.2: Import Required Libraries (one cell)
│   │   └── All imports with comments
│   └── 1.3: Global Configuration (one cell - SSoT)
│       ├── RANDOM_SEED constant
│       ├── All hyperparameters in config dict
│       └── Validation of config
│
├── 🔧 Step 2: Model/Data Initialization
│   └── Load models, datasets with clear error handling
│
├── 🎓 Step 3: Core Task Execution
│   └── Main computational work with progress indicators
│
├── 🔮 Step 4: Analysis and Validation
│   ├── Validation with assertions
│   ├── Visualization with consistent style
│   └── Biological interpretation
│
├── 📚 Step 5: Reference Code (Optional)
│   └── Self-contained standalone example
│
└── 🎉 Summary Cell (Markdown)
    ├── What was learned
    ├── Next steps
    └── Links to related tutorials
```

---

## Code Standards

### 1. Random Seed Management (CRITICAL)

**ALWAYS set seeds in Section 1.3 using shared_utils:**

```python
# ❌ WRONG: Scattered seed setting
import random
random.seed(42)
np.random.seed(42)  # Forgotten!

# ✅ CORRECT: Centralized, comprehensive
from shared_utils import set_global_seed

RANDOM_SEED = 42  # Single source of truth
set_global_seed(RANDOM_SEED, verbose=True)
```

**Rationale**: Ensures bit-exact reproducibility across Python, NumPy, PyTorch (CPU+CUDA).

---

### 2. Configuration Management (SSoT)

**ALL hyperparameters in ONE config dict, validated:**

```python
# ❌ WRONG: Scattered magic numbers
model = load_model("gpt2")
train(model, lr=0.001, epochs=50, batch=8)  # Where did these come from?

# ✅ CORRECT: Single Source of Truth
from shared_utils import validate_config

# All configuration in one place
config = {
    "model_name": "yangheng/OmniGenome-52M",
    "learning_rate": 1e-3,
    "epochs": 50,
    "batch_size": 8,
    "max_length": 512,
    "random_seed": RANDOM_SEED,
}

# Validate schema
schema = {
    "model_name": str,
    "learning_rate": float,
    "epochs": int,
    "batch_size": int,
}
validate_config(config, schema)

# Use throughout notebook
model = load_model(config["model_name"])
train(model, **config)
```

**Rationale**: Makes experiments reproducible and easy to modify.

---

### 3. Import Organization

**Standard import order:**

```python
# 1. Standard library (alphabetical)
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party libraries (alphabetical within groups)
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 3. OmniGenBench imports (grouped by functionality)
from omnigenbench import (
    OmniModelForEmbedding,          # Models
    OmniDatasetForSequenceClassification,  # Data
    ClassificationMetric,           # Metrics
    AutoBench,                     # Workflows
)

# 4. Local utilities
from shared_utils import (
    set_global_seed,
    verify_environment,
    assert_shape,
)
```

**Rationale**: Consistent order improves readability and avoids import errors.

---

### 4. Path Handling (Platform-Independent)

**Use pathlib.Path, never hardcoded strings:**

```python
# ❌ WRONG: Platform-specific, hardcoded
data_path = "D:\\data\\sequences.fasta"  # Fails on Linux
model_dir = "../models"  # Ambiguous relative path

# ✅ CORRECT: Platform-independent, explicit
from shared_utils import resolve_data_path, get_notebook_root

notebook_root = get_notebook_root()
data_path = resolve_data_path("data/sequences.fasta", create_if_missing=False)
model_dir = notebook_root / "models"
model_dir.mkdir(exist_ok=True)
```

**Rationale**: Works on Windows, Linux, macOS without modification.

---

### 5. Error Handling and Validation

**Fail fast with informative messages:**

```python
# ❌ WRONG: Silent failures
try:
    data = load_data(path)
except:
    data = None  # What went wrong?

# ✅ CORRECT: Explicit validation with assertions
from shared_utils import assert_shape

data = load_data(path)
assert data is not None, f"Failed to load data from {path}"
assert_shape(data, (-1, 768), "data embeddings")

# Validate ranges
assert 0 < config["learning_rate"] < 1, "learning_rate must be in (0, 1)"
```

**Rationale**: Catch errors early with clear diagnostics.

---

### 6. Progress Indicators

**Always show progress for long operations:**

```python
# ❌ WRONG: Black box computation
for seq in sequences:
    process(seq)  # User has no idea how long this takes

# ✅ CORRECT: Clear progress indication
from tqdm import tqdm

for seq in tqdm(sequences, desc="Processing sequences"):
    process(seq)
```

**Rationale**: Users know the notebook is working and can estimate completion time.

---

### 7. Visualization Standards

**Consistent matplotlib configuration:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Standard configuration (once per notebook)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Standard figure template
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
ax.plot(x, y, linewidth=2, alpha=0.7)
ax.set_xlabel("X Label", fontsize=12)
ax.set_ylabel("Y Label", fontsize=12)
ax.set_title("Clear Title", fontsize=14, pad=15)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()
```

**Rationale**: Consistent aesthetics across all tutorials.

---

### 8. Documentation Standards

**Every code cell has a preceding markdown cell explaining:**
- What this cell does
- Why it's necessary
- What output to expect
- Key parameters/concepts

**Example:**

```markdown
### 4.2: Dimensionality Reduction and Visualization

High-dimensional embeddings (768-D) are difficult to visualize. Let's use PCA 
and t-SNE to project them into 2D space while preserving the most important 
relationships.

**Expected Output**: 
- PCA explains ~60% variance with first 50 components
- t-SNE plot shows functional RNA groups clustering together
```

---

## Terminology Consistency

### Standard Terms Across All Notebooks

| Concept | Standard Term | ❌ Avoid |
|---------|--------------|----------|
| Pre-trained model | Genomic foundation model | Base model, pretrained LLM |
| Vector representation | Embedding | Feature, encoding |
| Fine-tuning process | Supervised fine-tuning | Transfer learning, retraining |
| Validation metric | Evaluation metric | Scoring function, performance measure |
| Random seed | `RANDOM_SEED` constant | seed, random_state (variable) |
| Configuration | `config` dictionary | params, settings, hyperparams |
| Device (GPU/CPU) | `device` from `torch.device()` | gpu, cuda_device |

---

## Windows Compatibility Rules

OmniGenBench runs on Windows by default. Follow these rules:

### ❌ Avoid Unicode Emoji in Print Statements

```python
# ❌ WRONG: Causes encoding errors on Windows terminals
print("🎯 Model loaded successfully!")

# ✅ CORRECT: Use text markers
print("[INFO] Model loaded successfully!")
print("[SUCCESS] Training completed!")
print("[ERROR] File not found!")
```

**Exception**: Markdown cells can use emoji freely (they render in browser).

### ✅ Use Forward Slashes in Paths

```python
# ✅ CORRECT: Always works
path = "examples/data/sequences.fasta"
path = Path("examples") / "data" / "sequences.fasta"

# ❌ AVOID: Windows-specific
path = "examples\\data\\sequences.fasta"
```

---

## Testing Requirements

### Every Tutorial Must Pass:

1. **Restart & Run All**: Complete execution without manual intervention
2. **Determinism Check**: Running twice produces identical outputs
3. **Import Test**: All imports succeed without internet (after first run)
4. **GPU Optional**: Must work on CPU-only machines
5. **Assertion Pass**: All `assert` statements succeed

### Validation Test Template

```python
# Final validation cell (include in every tutorial)
print("=" * 70)
print("VALIDATION CHECKS")
print("=" * 70)

# 1. Check reproducibility
assert RANDOM_SEED == 42, "Random seed was modified"

# 2. Check outputs have expected properties
assert_shape(embeddings, (len(sequences), 768), "embeddings")
assert embeddings.min() > -10 and embeddings.max() < 10, "Embedding values out of range"

# 3. Check saved artifacts exist
assert model_dir.exists(), f"Model directory not created: {model_dir}"

print("[SUCCESS] All validation checks passed!")
print("=" * 70)
```

---

## Shared Utilities API Reference

All notebooks should use `examples/shared_utils.py`:

### Core Functions

```python
from shared_utils import (
    # Environment Setup
    setup_notebook_environment,  # One-line setup (RECOMMENDED)
    set_global_seed,            # Set random seeds
    verify_environment,         # Check dependencies
    
    # Path Resolution
    get_notebook_root,          # Get current notebook directory
    get_project_root,           # Get OmniGenBench root
    resolve_data_path,          # Resolve relative paths
    
    # Validation
    validate_config,            # Validate config dictionaries
    assert_shape,               # Assert tensor shapes
    assert_close,               # Assert numeric closeness
    
    # Safe Imports
    safe_import,                # Import with helpful error messages
)
```

### Quick Start Template

```python
# Cell 1: Minimal setup
from shared_utils import setup_notebook_environment

env = setup_notebook_environment(
    seed=42,
    required_packages=['omnigenbench', 'torch', 'transformers'],
    verbose=True
)

# Cell 2: Your actual imports
from omnigenbench import OmniModelForEmbedding
# ... rest of tutorial
```

---

## Migration Guide

### Converting Old Notebooks to New Standard

1. **Add shared_utils import** to first code cell
2. **Replace scattered seed setting** with `set_global_seed(RANDOM_SEED)`
3. **Consolidate all magic numbers** into `config` dictionary
4. **Replace print() emoji** with `[INFO]` style markers
5. **Add validation assertions** at end of each major section
6. **Test "Restart & Run All"** to ensure reproducibility

### Example Refactoring

**Before:**
```python
import torch
model = torch.load("model.pt")
lr = 0.001  # Where does this come from?
train(model, lr)
```

**After:**
```python
from shared_utils import set_global_seed, resolve_data_path

RANDOM_SEED = 42
set_global_seed(RANDOM_SEED)

config = {
    "model_path": "model.pt",
    "learning_rate": 1e-3,
}

model_path = resolve_data_path(config["model_path"])
model = torch.load(model_path)
train(model, lr=config["learning_rate"])
```

---

## FAQ

### Q: Can I use different random seeds for different experiments?

**A**: Yes, but define them in the `config` dict:

```python
config = {
    "random_seed_data": 42,    # For data shuffling
    "random_seed_model": 123,  # For model initialization
}
```

### Q: What if my tutorial needs GPU?

**A**: Check availability and fail gracefully:

```python
env = setup_notebook_environment(check_gpu=True)

if not env['cuda_available']:
    print("[WARNING] GPU not available. This tutorial may be slow on CPU.")
    print("  Consider using Google Colab for free GPU access.")
```

### Q: How do I handle large files that can't be in the repo?

**A**: Provide download instructions and validate:

```python
data_path = resolve_data_path("data/large_file.h5")

if not data_path.exists():
    print("[ERROR] Large file not found. Download from:")
    print("  https://huggingface.co/datasets/yangheng/...")
    print(f"  Save to: {data_path}")
    raise FileNotFoundError(f"Required file: {data_path}")
```

---

## Change Log

- **2024-11**: Initial standard established
- Standards may evolve based on community feedback

---

## Enforcement

All pull requests must pass:
1. `pytest tests/test_notebook_reproducibility.py` (validates structure)
2. Manual review of "Restart & Run All" execution
3. Code review for adherence to these standards

**Maintainer**: YANG, HENG (@yangheng95)
