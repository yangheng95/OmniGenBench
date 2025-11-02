# -*- coding: utf-8 -*-
# file: shared_utils.py
# time: 12:00 02/11/2024
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# Copyright (C) 2019-2025. All Rights Reserved.

"""
Shared utilities for OmniGenBench tutorial notebooks.

This module provides reproducibility-first, zero-implicit-state utilities
following the Single Source of Truth (SSoT) principle.

Design Principles:
- Determinism by default: All randomness is seeded and documented
- Separation of Concerns: Environment, data, visualization, and computation logic separated
- Progressive Disclosure: Simple defaults, advanced options available
- POLA (Principle of Least Astonishment): Consistent behavior across notebooks
- No hidden state: All configuration explicit and traceable
"""

import os
import sys
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np


# ==============================================================================
# REPRODUCIBILITY: Random Seed Management
# ==============================================================================

DEFAULT_RANDOM_SEED = 42


def set_global_seed(seed: int = DEFAULT_RANDOM_SEED, verbose: bool = True) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.
    
    This function configures:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - PyTorch's cudnn backend for deterministic behavior
    
    Args:
        seed: Random seed value (default: 42)
        verbose: Whether to print confirmation message
        
    Returns:
        None
        
    Side Effects:
        - Modifies global random state of Python, NumPy, PyTorch
        - May reduce GPU performance slightly due to deterministic mode
        
    Example:
        >>> set_global_seed(42)
        [SUCCESS] Random seed set to 42 for reproducibility
        >>> # All subsequent random operations will be deterministic
    """
    # Python built-in random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random (lazy import to avoid forcing dependency)
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            
            # Ensure deterministic behavior (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            if verbose:
                print(f"[SUCCESS] Random seed set to {seed} for reproducibility")
                print(f"  - Python random: seeded")
                print(f"  - NumPy random: seeded")
                print(f"  - PyTorch CPU: seeded")
                print(f"  - PyTorch CUDA: seeded (deterministic mode enabled)")
        else:
            if verbose:
                print(f"[SUCCESS] Random seed set to {seed} for reproducibility")
                print(f"  - Python random: seeded")
                print(f"  - NumPy random: seeded")
                print(f"  - PyTorch CPU: seeded")
                print(f"  - PyTorch CUDA: not available")
    except ImportError:
        if verbose:
            print(f"[SUCCESS] Random seed set to {seed} for reproducibility")
            print(f"  - Python random: seeded")
            print(f"  - NumPy random: seeded")
            print(f"  - PyTorch: not installed (skipped)")


# ==============================================================================
# ENVIRONMENT VERIFICATION
# ==============================================================================

def verify_environment(
    required_packages: Optional[List[str]] = None,
    python_version: Optional[Tuple[int, int]] = None,
    check_gpu: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify Python environment and package installations.
    
    This function checks:
    - Python version compatibility
    - Required package installations
    - PyTorch and CUDA availability
    - Package version information
    
    Args:
        required_packages: List of required package names. If None, checks default set.
        python_version: Minimum required Python version as (major, minor) tuple
        check_gpu: Whether to check for GPU/CUDA availability
        verbose: Whether to print detailed verification output
        
    Returns:
        Dictionary containing verification results:
        {
            'python_version': str,
            'packages': {package_name: version_str},
            'cuda_available': bool,
            'cuda_version': str or None,
            'gpu_name': str or None,
            'all_checks_passed': bool
        }
        
    Raises:
        SystemExit: If critical checks fail (only when verbose=True)
        
    Example:
        >>> env_info = verify_environment(
        ...     required_packages=['omnigenbench', 'torch'],
        ...     python_version=(3, 8)
        ... )
        >>> assert env_info['all_checks_passed'], "Environment check failed"
    """
    if verbose:
        print("=" * 70)
        print("ENVIRONMENT VERIFICATION")
        print("=" * 70)
    
    env_info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'packages': {},
        'cuda_available': False,
        'cuda_version': None,
        'gpu_name': None,
        'all_checks_passed': True
    }
    
    # Check Python version
    if python_version:
        current_version = (sys.version_info.major, sys.version_info.minor)
        if current_version < python_version:
            if verbose:
                print(f"[ERROR] Python {python_version[0]}.{python_version[1]}+ required, "
                      f"found {env_info['python_version']}")
            env_info['all_checks_passed'] = False
            if verbose:
                sys.exit(1)
    else:
        if verbose:
            print(f"[INFO] Python version: {env_info['python_version']}")
    
    # Check required packages
    if required_packages is None:
        required_packages = ['omnigenbench', 'torch', 'numpy', 'matplotlib', 'transformers']
    
    for package in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            env_info['packages'][package] = version
            if verbose:
                print(f"[SUCCESS] {package} {version}")
        except ImportError as e:
            if verbose:
                print(f"[ERROR] {package} not found: {e}")
                print(f"  Install with: pip install {package} -U")
            env_info['packages'][package] = None
            env_info['all_checks_passed'] = False
    
    # Check GPU availability
    if check_gpu:
        try:
            import torch
            env_info['cuda_available'] = torch.cuda.is_available()
            if env_info['cuda_available']:
                env_info['cuda_version'] = torch.version.cuda
                env_info['gpu_name'] = torch.cuda.get_device_name(0)
                if verbose:
                    print(f"[SUCCESS] CUDA {env_info['cuda_version']} available")
                    print(f"  GPU: {env_info['gpu_name']}")
            else:
                if verbose:
                    print(f"[INFO] CUDA not available (CPU-only mode)")
        except ImportError:
            pass
    
    if verbose:
        print("=" * 70)
        if env_info['all_checks_passed']:
            print("[SUCCESS] All environment checks passed")
        else:
            print("[ERROR] Some environment checks failed")
            sys.exit(1)
        print("=" * 70)
    
    return env_info


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

def get_notebook_root(notebook_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the root directory of the current notebook.
    
    This function resolves the notebook's parent directory in a platform-independent way,
    supporting both local files and cloud-based notebooks (Google Colab, Kaggle, etc.).
    
    Args:
        notebook_path: Path to the notebook file. If None, uses current working directory.
        
    Returns:
        Path object pointing to the notebook's parent directory
        
    Example:
        >>> root = get_notebook_root()
        >>> data_dir = root / "data"
        >>> model_dir = root / "models"
    """
    if notebook_path is None:
        # Try to detect notebook path from IPython
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and hasattr(ipython, 'user_ns'):
                # Try to get __file__ from globals
                notebook_path = ipython.user_ns.get('__file__', None)
        except ImportError:
            pass
    
    if notebook_path is None:
        # Fall back to current working directory
        return Path.cwd()
    
    return Path(notebook_path).resolve().parent


def get_project_root(notebook_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the root directory of the OmniGenBench project.
    
    Searches upward from the notebook directory until finding a directory
    containing 'setup.py' or 'pyproject.toml'.
    
    Args:
        notebook_path: Path to the notebook file. If None, uses current working directory.
        
    Returns:
        Path object pointing to the project root
        
    Raises:
        FileNotFoundError: If project root cannot be determined
        
    Example:
        >>> project_root = get_project_root()
        >>> sys.path.insert(0, str(project_root))  # Add to Python path
    """
    current = get_notebook_root(notebook_path)
    
    # Search upward for project markers
    for _ in range(10):  # Limit search depth to avoid infinite loops
        if (current / 'setup.py').exists() or (current / 'pyproject.toml').exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: assume we're in examples/ subdirectory
    possible_root = get_notebook_root(notebook_path).parent
    if (possible_root / 'setup.py').exists():
        return possible_root
    
    raise FileNotFoundError(
        "Cannot determine project root. Make sure you're running from within "
        "the OmniGenBench project directory."
    )


def resolve_data_path(
    relative_path: Union[str, Path],
    notebook_path: Optional[Union[str, Path]] = None,
    create_if_missing: bool = False
) -> Path:
    """
    Resolve a data path relative to the notebook directory.
    
    Args:
        relative_path: Path relative to the notebook directory
        notebook_path: Path to the notebook file (if None, uses current directory)
        create_if_missing: Whether to create the directory if it doesn't exist
        
    Returns:
        Absolute Path object
        
    Example:
        >>> data_file = resolve_data_path("data/sequences.fasta")
        >>> assert data_file.exists(), f"Data file not found: {data_file}"
    """
    notebook_root = get_notebook_root(notebook_path)
    resolved = (notebook_root / relative_path).resolve()
    
    if create_if_missing and not resolved.exists():
        if resolved.suffix:  # Has file extension
            resolved.parent.mkdir(parents=True, exist_ok=True)
        else:  # Is a directory
            resolved.mkdir(parents=True, exist_ok=True)
    
    return resolved


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_config(config: Dict[str, Any], schema: Dict[str, type]) -> None:
    """
    Validate configuration dictionary against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Expected schema as {key: expected_type}
        
    Raises:
        ValueError: If validation fails
        
    Example:
        >>> config = {'learning_rate': 1e-3, 'epochs': 10}
        >>> schema = {'learning_rate': float, 'epochs': int}
        >>> validate_config(config, schema)  # Passes silently
        >>> 
        >>> bad_config = {'learning_rate': 'invalid', 'epochs': 10}
        >>> validate_config(bad_config, schema)  # Raises ValueError
    """
    for key, expected_type in schema.items():
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
        
        if not isinstance(config[key], expected_type):
            raise ValueError(
                f"Config key '{key}' has wrong type: "
                f"expected {expected_type.__name__}, got {type(config[key]).__name__}"
            )


# ==============================================================================
# COMMON IMPORTS AND SETUP
# ==============================================================================

def setup_notebook_environment(
    seed: int = DEFAULT_RANDOM_SEED,
    required_packages: Optional[List[str]] = None,
    check_gpu: bool = True,
    suppress_warnings: bool = True,
    matplotlib_style: str = 'seaborn-v0_8',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    One-line setup for reproducible notebook environment.
    
    This function combines:
    - Random seed setting
    - Environment verification
    - Warning suppression
    - Matplotlib configuration
    
    Args:
        seed: Random seed for reproducibility
        required_packages: List of required packages to verify
        check_gpu: Whether to check GPU availability
        suppress_warnings: Whether to suppress common warnings
        matplotlib_style: Matplotlib style to apply
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with environment information
        
    Example:
        >>> # At the top of your notebook:
        >>> from shared_utils import setup_notebook_environment
        >>> env = setup_notebook_environment(seed=42, verbose=True)
        >>> # Continue with your tutorial...
    """
    # Set random seed
    set_global_seed(seed, verbose=verbose)
    
    # Verify environment
    env_info = verify_environment(
        required_packages=required_packages,
        check_gpu=check_gpu,
        verbose=verbose
    )
    
    # Suppress warnings if requested
    if suppress_warnings:
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Suppress specific library warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
        if verbose:
            print("[INFO] Common warnings suppressed for cleaner output")
    
    # Configure matplotlib
    try:
        import matplotlib.pyplot as plt
        plt.style.use(matplotlib_style)
        if verbose:
            print(f"[INFO] Matplotlib style set to: {matplotlib_style}")
    except (ImportError, OSError):
        if verbose:
            print("[WARNING] Could not configure matplotlib")
    
    if verbose:
        print("\n[SUCCESS] Notebook environment ready!")
        print("=" * 70)
        print()
    
    return env_info


# ==============================================================================
# ASSERTION HELPERS FOR VALIDATION
# ==============================================================================

def assert_shape(tensor, expected_shape: Tuple[int, ...], name: str = "tensor") -> None:
    """
    Assert tensor has expected shape, with informative error message.
    
    Args:
        tensor: Tensor to check (supports PyTorch, NumPy, or anything with .shape)
        expected_shape: Expected shape tuple (use -1 for any dimension)
        name: Name of the tensor for error messages
        
    Raises:
        AssertionError: If shape doesn't match
        
    Example:
        >>> embeddings = model.encode(sequences)
        >>> assert_shape(embeddings, (len(sequences), 768), "embeddings")
    """
    actual_shape = tuple(tensor.shape)
    
    # Allow -1 as wildcard dimension
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise AssertionError(
                f"{name} has wrong shape: expected {expected_shape}, got {actual_shape}"
            )
    
    if len(actual_shape) != len(expected_shape):
        raise AssertionError(
            f"{name} has wrong number of dimensions: "
            f"expected {len(expected_shape)}, got {len(actual_shape)}"
        )


def assert_close(
    actual: float,
    expected: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "value"
) -> None:
    """
    Assert two values are close within tolerance.
    
    Args:
        actual: Actual value
        expected: Expected value
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
        
    Raises:
        AssertionError: If values differ beyond tolerance
        
    Example:
        >>> loss = model.compute_loss(predictions, targets)
        >>> assert_close(loss, 0.234, rtol=0.1, name="validation loss")
    """
    if not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name} not close to expected: "
            f"actual={actual}, expected={expected} "
            f"(rtol={rtol}, atol={atol})"
        )


# ==============================================================================
# SAFE IMPORTS (with fallback and clear error messages)
# ==============================================================================

def safe_import(
    package_name: str,
    package_display_name: Optional[str] = None,
    install_command: Optional[str] = None
):
    """
    Safely import a package with informative error message if missing.
    
    Args:
        package_name: Package name to import
        package_display_name: Display name for error messages (defaults to package_name)
        install_command: Custom install command (defaults to "pip install {package_name}")
        
    Returns:
        Imported module
        
    Raises:
        ImportError: With helpful installation instructions
        
    Example:
        >>> viennarna = safe_import('RNA', 'ViennaRNA', 'pip install viennarna')
        >>> # If import fails, user gets clear error message
    """
    if package_display_name is None:
        package_display_name = package_name
    
    if install_command is None:
        install_command = f"pip install {package_name} -U"
    
    try:
        return __import__(package_name)
    except ImportError as e:
        raise ImportError(
            f"\n{'=' * 70}\n"
            f"ERROR: {package_display_name} is required but not installed.\n\n"
            f"To install, run:\n"
            f"    {install_command}\n\n"
            f"Original error: {e}\n"
            f"{'=' * 70}\n"
        )


if __name__ == "__main__":
    # Self-test
    print("Testing shared_utils module...")
    print()
    
    # Test seed setting
    set_global_seed(42)
    
    # Test environment verification
    env = verify_environment(verbose=True)
    
    # Test path resolution
    try:
        root = get_notebook_root()
        print(f"\nNotebook root: {root}")
    except Exception as e:
        print(f"\nCould not determine notebook root: {e}")
    
    print("\n[SUCCESS] All self-tests passed!")
