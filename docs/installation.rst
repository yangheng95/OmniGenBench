############
Installation
############


Get started with OmniGenBench by installing it in your preferred environment. We recommend using a Python virtual environment to manage dependencies. Python 3.10 or higher is required.


**************************
Installation with pip
**************************

The simplest way to install OmniGenBench is via ``pip``. This will install the latest stable release along with its core dependencies.
We recommend installing OmniGenBench in a dedicated conda environment to manage its dependencies effectively.

Create and activate a new Conda environment:

.. code-block:: bash

   conda create -n omnigen_env python==3.12.0
   conda activate omnigen_env

Now, install OmniGenBench from PyPI with the following command:

.. code-block:: bash

   pip install omnigenbench

This command will install OmniGenBench along with its essential dependencies, such as ``torch``, ``transformers``, and ``accelerate``.

To verify the installation was successful, you can run:

.. code-block:: bash

   python -c "import omnigenbench; print(omnigenbench.__version__)"


*********************************
Installation from source (Editable)
*********************************

Installing from the source is recommended if you plan to contribute to the project or need the very latest, unreleased features. This is often called an "editable" install.

.. code-block:: bash

   # Clone the repository from GitHub
   git clone https://github.com/COLA-Laboratory/OmniGenBench.git
   cd OmniGenBench

   # Install in editable mode with all development dependencies
   pip install -e .

This approach links the installed package directly to your local source code, so any changes you make will be immediately effective without needing to reinstall.

.. _core-dependencies:

*******************
Core Dependencies
*******************

OmniGenBench is built on top of the modern Python data science and AI ecosystem. The installation process will automatically handle these dependencies, but for reference, here are the major ones:

*   **PyTorch**: ``torch>=2.6.0`` is required for all model and tensor operations.
*   **Hugging Face Ecosystem**:
    *   ``transformers>=4.46.0`` for model architectures and backbones.
    *   ``accelerate`` for seamless distributed training and inference.
    *   ``peft`` for Parameter-Efficient Fine-Tuning.
*   **Core Utilities**: ``findfile``, ``autocuda``, ``metric-visualizer``, ``packaging``, ``dill``.
*   **Data Handling**: ``pandas``, ``scikit-learn``.
*   **Specialized Tools**: ``viennarna``, ``gitpython``.


***********************
Common Troubleshooting
***********************

.. tip::
   Using a package manager like `Anaconda <https://www.anaconda.com/products/distribution>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ can greatly simplify managing complex dependencies, especially CUDA. This is highly recommended for all users, particularly on Windows.

**CUDA and PyTorch Issues**

If you encounter errors related to CUDA or GPU detection, it usually means there is a mismatch between your NVIDIA driver, your CUDA Toolkit version, and the PyTorch build.

1.  First, check your CUDA version with ``nvidia-smi``.
2.  Then, visit the `PyTorch official website <https://pytorch.org/get-started/locally/>`_ to find the exact ``pip`` or ``conda`` command that matches your system's CUDA version.

**Version Conflicts**

If you face issues with package versions (e.g., ``transformers`` or ``accelerate``), try creating a fresh virtual environment or forcing an upgrade of the conflicting package:

.. code-block:: bash

   pip install --upgrade transformers accelerate

For any other issues, please feel free to `open an issue on our GitHub repository <https://github.com/yangheng95/OmniGenBench/issues>`_.
