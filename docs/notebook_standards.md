# OmniGenBench Notebook Standardization Guide

**Version**: 2.0  
**Last Updated**: November 2, 2025  
**Status**: ENFORCED via automated tooling

---

## 🎯 Design Philosophy

### Core Principles

1. **Reproducibility First**: Every notebook must produce identical results across executions
2. **Zero Hidden State**: No implicit dependencies on execution order or external files
3. **Self-Contained**: All utilities embedded directly (no `shared_utils.py`)
4. **Single Source of Truth (SSoT)**: Configuration centralized in one cell
5. **Fail Fast**: Validation at configuration time, not runtime
6. **Restart & Run All**: This is the ONLY supported execution mode

---

## 📋 Mandatory Structure

### Cell Organization (Strict Order)

Every tutorial notebook MUST follow this exact structure:

```
1. Title & Overview (Markdown)
   - Tutorial purpose and learning objectives
   - Visual workflow diagram (mermaid)
   - Prerequisites and estimated time

2. Environment Setup (Code)
   - Package installation (commented out)
   - Embedded utility functions
   - Environment verification
   - Reproducibility setup (seeds)

3. Library Imports (Code)
   - Standard library
   - Third-party packages
   - OmniGenBench components
   - Import verification

4. Global Configuration (Code - SSoT)
   - ALL parameters in one cell
   - Configuration validation
   - Schema enforcement
   - Print configuration summary

5. Tutorial Steps (Markdown + Code pairs)
   - Step N: Description (Markdown)
   - Step N: Implementation (Code)
   - Repeat for each step

6. Summary & Validation (Markdown)
   - Learning outcomes recap
   - Reproducibility check
   - Next steps and resources
```

---

## 🔒 Mandatory Patterns

### 1. Embedded Utilities (CRITICAL)

**❌ FORBIDDEN:**
```python
from shared_utils import set_global_seed  # NO EXTERNAL DEPENDENCIES
```

**✅ REQUIRED:**
```python
# =============================================================================
# EMBEDDED UTILITIES (Reproducibility Framework)
# =============================================================================
# These functions are embedded directly to ensure notebook independence

def set_global_seed(seed: int, verbose: bool = True):
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    Args:
        seed: Integer seed value
        verbose: Whether to print confirmation
        
    Returns:
        dict: Environment information
    """
    import numpy as np
    import torch
    
    # Set seeds for all random number generators
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if verbose:
        print(f"[SEED] Global seed set to: {seed}")
    
    return {"seed": seed, "deterministic": True}

# ... other utilities embedded here ...
```

### 2. Single Source of Truth (SSoT) Configuration

**ALL** configurable parameters MUST be in ONE cell:

```python
# =============================================================================
# GLOBAL CONFIGURATION (Single Source of Truth)
# =============================================================================

# Random Seed (CRITICAL for Reproducibility)
RANDOM_SEED = 42

# Set seeds immediately
set_global_seed(RANDOM_SEED, verbose=True)

# Model Configuration
model_config = {
    "model_name": "yangheng/OmniGenome-52M",
    "max_length": 512,
    "batch_size": 16,
    "use_fp16": True,
}

# Analysis Configuration
analysis_config = {
    "n_components": 50,
    "n_clusters": 4,
    "random_state": RANDOM_SEED,  # Consistency
}

# Visualization Configuration
viz_config = {
    "figsize": (12, 8),
    "dpi": 100,
    "cmap": "viridis",
}

# =============================================================================
# Configuration Validation (Fail Fast)
# =============================================================================

def validate_config(config: dict, schema: dict):
    """Validate configuration against schema."""
    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Config key '{key}' has type {type(config[key]).__name__}, "
                f"expected {expected_type.__name__}"
            )

# Validate all configs
model_schema = {
    "model_name": str,
    "max_length": int,
    "batch_size": int,
    "use_fp16": bool,
}
validate_config(model_config, model_schema)

# Print configuration summary
print("\n" + "=" * 70)
print("CONFIGURATION SUMMARY (Single Source of Truth)")
print("=" * 70)
print(f"\n[SEED] Random seed: {RANDOM_SEED}")
print(f"\n[MODEL] {model_config}")
print(f"\n[ANALYSIS] {analysis_config}")
print(f"\n[VIZ] {viz_config}")
print("=" * 70 + "\n")
```

### 3. Reproducibility Guarantees

Every notebook MUST include:

```python
# At the start (in environment setup cell)
import random
import os
import numpy as np
import torch

def set_global_seed(seed: int, verbose: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if verbose:
        print(f"[SEED] Global seed set to: {seed}")
        print("  - Python random, NumPy, PyTorch (CPU & CUDA)")
        print("  - PyTorch backends configured for determinism")
    
    return {"seed": seed, "deterministic": True}
```

### 4. Environment Verification

Every notebook MUST verify the environment:

```python
def verify_environment(required_packages: list, check_gpu: bool = True):
    """Verify packages and GPU availability."""
    import sys
    
    print(f"\n{'='*70}")
    print("ENVIRONMENT VERIFICATION")
    print(f"{'='*70}")
    print(f"Python version: {sys.version.split()[0]}")
    
    missing_packages = []
    for pkg in required_packages:
        try:
            module = __import__(pkg)
            version = getattr(module, '__version__', 'unknown')
            print(f"[OK] {pkg:20s} v{version}")
        except ImportError:
            missing_packages.append(pkg)
            print(f"[MISSING] {pkg:20s}")
    
    if missing_packages:
        raise ImportError(
            f"Missing packages: {', '.join(missing_packages)}\n"
            f"Install with: pip install {' '.join(missing_packages)}"
        )
    
    if check_gpu:
        import torch
        gpu_available = torch.cuda.is_available()
        print(f"\n[GPU] Available: {gpu_available}")
        if gpu_available:
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
    
    print(f"{'='*70}\n")
```

---

## 🚫 Anti-Patterns (Strictly Forbidden)

### 1. External Dependencies

**❌ FORBIDDEN:**
```python
from shared_utils import setup_environment
from ../common/helpers import process_data
import sys; sys.path.append('..')
```

**Reason**: Breaks "Restart & Run All" guarantee. Notebooks become order-dependent and fragile.

### 2. Hidden Magic Numbers

**❌ FORBIDDEN:**
```python
# Cell 5
embeddings = model.encode(seq, max_length=512)

# Cell 10
reduced = pca.fit_transform(embeddings[:, :512])  # Magic 512!
```

**✅ REQUIRED:**
```python
# Configuration cell
CONFIG = {
    "max_length": 512,
}

# Later cells
embeddings = model.encode(seq, max_length=CONFIG["max_length"])
reduced = pca.fit_transform(embeddings[:, :CONFIG["max_length"]])
```

### 3. Mutable Global State

**❌ FORBIDDEN:**
```python
# Cell 3
results = []

# Cell 7
results.append(accuracy)  # Depends on execution order!
```

**✅ REQUIRED:**
```python
# Each cell is self-contained or explicitly passes data
def compute_metrics(predictions, labels):
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

results = compute_metrics(predictions, labels)
```

### 4. Implicit Package Installations

**❌ FORBIDDEN:**
```python
!pip install omnigenbench  # Uncomment if needed
```

**✅ REQUIRED:**
```python
# Install required packages (uncomment if running for first time)
# !pip install omnigenbench torch transformers -U
```

With clear instructions and always commented out by default.

---

## 📊 Validation Checklist

Before submitting any notebook, verify:

### Structural Requirements
- [ ] Title cell with learning objectives and prerequisites
- [ ] Environment setup cell with embedded utilities
- [ ] Library imports cell with verification
- [ ] Configuration cell with SSoT pattern
- [ ] Tutorial steps follow Markdown → Code pairs
- [ ] Summary cell with reproducibility validation

### Reproducibility Requirements
- [ ] `RANDOM_SEED` defined in configuration cell
- [ ] `set_global_seed()` called immediately after defining seed
- [ ] All random operations use the seed (NumPy, PyTorch, sklearn, etc.)
- [ ] Configuration validation with fail-fast schema checking

### Independence Requirements
- [ ] No imports from `shared_utils.py` or any external files
- [ ] All utility functions embedded directly in notebook
- [ ] No `sys.path` manipulation
- [ ] No relative imports outside the notebook

### Clarity Requirements
- [ ] All magic numbers moved to configuration cell
- [ ] Configuration printed at runtime for transparency
- [ ] Clear section headers with numbering
- [ ] Code cells have explanatory comments

### Execution Requirements
- [ ] `Kernel → Restart & Run All` completes without errors
- [ ] Results are identical across multiple runs (same seed)
- [ ] No "File not found" errors (all data embedded or downloaded)
- [ ] No GPU-only requirements (CPU fallback always available)

---

## 🔧 Automated Validation

The `dev/standardize_notebooks.py` script enforces these standards:

```bash
# Check all notebooks
python dev/standardize_notebooks.py --check

# Fix violations automatically
python dev/standardize_notebooks.py --fix

# Validate specific notebook
python dev/standardize_notebooks.py --check examples/genomic_embeddings/RNA_Embedding_Tutorial.ipynb
```

### Validation Rules

1. **No external imports**: Detects `from shared_utils import ...`
2. **SSoT presence**: Ensures configuration cell exists
3. **Seed usage**: Verifies `RANDOM_SEED` is defined and used
4. **Embedded utilities**: Checks that required functions are present
5. **Structure compliance**: Validates cell order matches template

---

## 📚 Templates

### Minimal Tutorial Template

```python
# Cell 1: Title (Markdown)
"""
# Tutorial: [Topic Name]

**Learning Objectives**:
- Objective 1
- Objective 2

**Prerequisites**: Python 3.8+, OmniGenBench
**Estimated Time**: 20 minutes
"""

# Cell 2: Environment Setup (Code)
"""
# Install packages (uncomment if needed)
# !pip install omnigenbench torch -U

# Embedded utilities
def set_global_seed(seed: int):
    # ... full implementation ...
    pass

def verify_environment(required_packages: list):
    # ... full implementation ...
    pass

# Execute setup
env_info = setup_notebook_environment(seed=42)
"""

# Cell 3: Imports (Code)
"""
import torch
import numpy as np
from omnigenbench import OmniModelForEmbedding

print("[SUCCESS] Imports completed")
"""

# Cell 4: Configuration (Code - SSoT)
"""
RANDOM_SEED = 42
set_global_seed(RANDOM_SEED)

config = {
    "model_name": "yangheng/OmniGenome-52M",
    "batch_size": 16,
}

# Validate
assert isinstance(config["batch_size"], int)
print(f"[CONFIG] {config}")
"""

# Cell 5+: Tutorial steps (Markdown + Code pairs)
# ...
```

---

## 🎓 Educational Philosophy

### Why These Standards?

1. **Students learn by example**: If tutorials have implicit dependencies, students will copy that anti-pattern
2. **Research reproducibility crisis**: Notebooks that "work on my machine" are not science
3. **Maintainability**: Shared utilities become a maintenance nightmare when they change
4. **Pedagogical clarity**: Self-contained notebooks force authors to be explicit about all dependencies

### What Students Should Learn

- **Explicit over implicit**: All dependencies and configurations visible
- **Validation over assumptions**: Check everything, fail fast
- **Reproducibility discipline**: Seeds, determinism, verification
- **Production readiness**: These patterns scale to real research projects

---

## 🚀 Migration Guide

### Migrating Existing Notebooks

1. **Identify external dependencies**:
   ```bash
   grep -r "from shared_utils" examples/
   grep -r "sys.path" examples/
   ```

2. **Extract and embed utilities**:
   - Copy function implementations directly into notebook
   - Add docstrings and type hints
   - Test in isolation

3. **Centralize configuration**:
   - Find all hardcoded parameters
   - Create SSoT configuration cell
   - Replace magic numbers with config references

4. **Add validation**:
   - Schema validation for configs
   - Environment verification
   - Reproducibility checks

5. **Test thoroughly**:
   ```python
   # Run twice, compare results
   jupyter nbconvert --execute --to notebook notebook.ipynb --output run1.ipynb
   jupyter nbconvert --execute --to notebook notebook.ipynb --output run2.ipynb
   # Results should be identical
   ```

---

## 📖 References

- [Ten Simple Rules for Reproducible Computational Research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285)
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Notebook%20Basics.html)
- [OmniGenBench Documentation](https://omnigenbench.readthedocs.io/)

---

**Version History**:
- v2.0 (2025-11-02): Complete rewrite with embedded utilities mandate
- v1.0 (2024): Initial standards document

**Maintained by**: YANG, HENG <hy345@exeter.ac.uk>
