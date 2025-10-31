############
Installation
############


Get started with OmniGenBench by installing it in your preferred environment. We strongly recommend using a dedicated Python virtual environment (via conda or venv) to manage dependencies and avoid conflicts. Python 3.10 or higher is required, with Python 3.12 recommended for optimal compatibility and performance.


**************************
Installation with pip
**************************

The simplest installation method uses ``pip`` to install the latest stable release from PyPI, along with all core dependencies required for basic functionality.

We recommend creating a dedicated conda environment to isolate OmniGenBench dependencies:

.. code-block:: bash

   # Create and activate new conda environment (Python 3.10+ required, 3.12 recommended)
   conda create -n omnigen_env python=3.12
   conda activate omnigen_env

Install OmniGenBench from PyPI:

.. code-block:: bash

   pip install omnigenbench

This command installs OmniGenBench along with essential dependencies: ``torch``, ``transformers``, ``accelerate``, and core bioinformatics libraries.

Verify successful installation:

.. code-block:: bash

   python -c "import omnigenbench; print(omnigenbench.__version__)"
   # Expected output: 0.3.23alpha (or current version)


*********************************
Installation from Source (Editable)
*********************************

Installing from source is recommended for contributors, developers, or users requiring unreleased features. This "editable" installation links the installed package directly to your local source code, so modifications take effect immediately without reinstallation.

.. code-block:: bash

   # Clone repository from GitHub
   git clone https://github.com/COLA-Laboratory/OmniGenBench.git
   cd OmniGenBench

   # Install in editable mode with all dependencies
   pip install -e .

With editable installation, any changes to Python files in the ``omnigenbench/`` directory are immediately reflected in the installed package, enabling rapid development and testing workflows.

*******************
Core Dependencies
*******************

OmniGenBench is built on the modern Python deep learning and bioinformatics ecosystem. The installation process automatically handles these core dependencies:

*   **PyTorch**: ``torch>=2.6.0`` for neural network operations and GPU acceleration
    
    - Ensure CUDA-compatible version for GPU support (check `pytorch.org <https://pytorch.org/>`_ for CUDA-specific installation)
    - CPU-only installation supported but significantly slower for large models

*   **HuggingFace Ecosystem**:
    
    - ``transformers>=4.46.0`` for transformer architectures and pre-trained model loading
    - ``accelerate>=0.21.0`` for distributed training across multiple GPUs/nodes with gradient accumulation
    - ``peft>=0.5.0`` for parameter-efficient fine-tuning (LoRA, prefix tuning, adapters)

*   **Bioinformatics Tools**: 
    
    - ``viennarna>=2.6.0`` for RNA secondary structure prediction via thermodynamic folding and free energy calculations
    - **Installation Note**: ViennaRNA requires system-level C libraries. Install via conda: ``conda install -c bioconda viennarna``

*   **Data Processing**: ``pandas``, ``scikit-learn``, ``scipy``, ``numpy`` for data manipulation, metrics, and numerical computation

*   **Visualization**: ``plotly``, ``matplotlib``, ``logomaker``, ``metric-visualizer>=0.9.6`` for result presentation and attention visualization

*   **Workflow Utilities**: ``findfile>=2.0.0``, ``autocuda>=0.16``, ``packaging``, ``dill``, ``gitpython`` for file discovery, GPU management, and serialization


***********************
Optional Dependencies
***********************

Some features require additional packages that are not installed by default:

*   **Web Interface**: ``gradio>=4.0.0`` for interactive web-based model demos
*   **File Format Support**: ``biopython`` for FASTA/FASTQ file parsing (auto-installed when needed)
*   **Advanced Visualization**: ``tabulate`` for formatted table output
*   **Development Tools**: ``pytest`` for running tests, ``dill`` for enhanced serialization

To install all optional dependencies:

.. code-block:: bash

   pip install omnigenbench[dev]


***********************
Common Troubleshooting
***********************

.. tip::
   Using a package manager like `Anaconda <https://www.anaconda.com/products/distribution>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ can greatly simplify managing complex dependencies, especially CUDA and ViennaRNA. This is highly recommended for all users, particularly on Windows.

**Windows-Specific Issues**

OmniGenBench is fully compatible with Windows, but be aware of these platform-specific considerations:

*   **Terminal Encoding**: If you see encoding errors or garbled output, ensure your terminal uses UTF-8 encoding:

    .. code-block:: bash
    
       # PowerShell
       [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
       
       # Git Bash (recommended for Windows users)
       export PYTHONIOENCODING=utf-8

*   **Path Separators**: Use forward slashes (``/``) or ``pathlib.Path`` objects for cross-platform compatibility:

    .. code-block:: python
    
       from pathlib import Path
       output_dir = Path("results/model")  # Works on all platforms

**ViennaRNA Installation**

ViennaRNA is required for RNA secondary structure prediction tasks. Installation varies by platform:

*   **Linux (Ubuntu/Debian)**:

    .. code-block:: bash
    
       sudo apt-get install python3-viennarna
       # Or via pip (may require build tools)
       pip install viennarna

*   **macOS**:

    .. code-block:: bash
    
       brew install viennarna
       pip install viennarna

*   **Windows**: ViennaRNA installation on Windows can be challenging. Recommended approach:

    .. code-block:: bash
    
       # Use conda (simplest method)
       conda install -c bioconda viennarna
       
       # Or use WSL2 (Windows Subsystem for Linux)
       wsl --install
       # Then follow Linux instructions inside WSL

*   **Docker Alternative**: For reproducible environments:

    .. code-block:: bash
    
       docker pull yangheng95/omnigenbench:latest

**CUDA and PyTorch Issues**

If you encounter errors related to CUDA or GPU detection, there is likely a mismatch between your NVIDIA driver, CUDA Toolkit version, and PyTorch build.

1.  Check your CUDA version with ``nvidia-smi``
2.  Visit the `PyTorch official website <https://pytorch.org/get-started/locally/>`_ to find the exact installation command matching your system's CUDA version
3.  Reinstall PyTorch with the correct CUDA version:

    .. code-block:: bash
    
       # Example for CUDA 11.8
       pip install torch --index-url https://download.pytorch.org/whl/cu118
       
       # Example for CUDA 12.1
       pip install torch --index-url https://download.pytorch.org/whl/cu121

4.  Verify installation:

    .. code-block:: python
    
       import torch
       print(f"CUDA available: {torch.cuda.is_available()}")
       print(f"CUDA version: {torch.version.cuda}")
       print(f"Device count: {torch.cuda.device_count()}")

**Memory Issues**

For large models or long sequences:

*   **Reduce batch size**: Use ``--batch-size 8`` or smaller
*   **Enable gradient checkpointing**: Trades compute for memory
*   **Use mixed precision**: ``--autocast`` flag for automatic FP16
*   **Monitor GPU memory**: ``nvidia-smi`` or ``watch -n 1 nvidia-smi``

**Import Errors**

If you encounter ``ModuleNotFoundError`` after installation:

1.  Verify installation: ``pip show omnigenbench``
2.  Check virtual environment activation: ``which python``
3.  Reinstall in clean environment:

    .. code-block:: bash
    
       conda create -n omnigen_fresh python=3.12
       conda activate omnigen_fresh
       pip install omnigenbench

**HuggingFace Hub Authentication**

For private models or high-rate API access:

.. code-block:: bash

   # Login with your HuggingFace token
   huggingface-cli login
   
   # Or set environment variable
   export HUGGINGFACE_TOKEN=your_token_here

**Version Conflicts**

If you face issues with package versions (e.g., ``transformers`` or ``accelerate``), try creating a fresh virtual environment or forcing an upgrade of the conflicting package:

.. code-block:: bash

   pip install --upgrade transformers accelerate

For any other issues, please feel free to `open an issue on our GitHub repository <https://github.com/yangheng95/OmniGenBench/issues>`_.
