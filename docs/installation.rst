.. Installation
.. ==================

.. This page describes **how to install OmniGenBench** and its dependencies.


.. Dependencies
.. ------------

.. OmniGenBench depends on the following major Python packages:

.. - findfile>=2.0.0
.. - autocuda>=0.16
.. - metric-visualizer>=0.9.6
.. - termcolor
.. - gitpython
.. - torch>=2.6.0
.. - pandas
.. - viennarna
.. - scikit-learn
.. - accelerate
.. - transformers>=4.46.0
.. - packaging
.. - peft
.. - dill


.. Basic Installation
.. ------------------

.. We recommend installing via `pip`:

.. .. code-block:: bash

..    pip install omnigenbench


.. Development Installation
.. ------------------------

.. Python 3.10 or higher is recommended.

.. To install development and testing dependencies:

.. .. code-block:: bash

..    git clone https://github.com/yangheng95/OmniGenBench.git
..    cd OmniGenBench
..    python setup.py install


.. Optional Dependencies
.. ---------------------

.. - accelerate *(for distributed training/inference)*  
.. - sphinx_rtd_theme *(recommended for building docs locally)*

.. Common Issues
.. -------------

.. - If you encounter CUDA/torch-related issues, make sure the correct version of PyTorch is installed.
.. - For `transformers` version conflicts, consider upgrading to the latest release.
.. - Windows users are strongly encouraged to use Anaconda or Miniconda environments.

.. For additional help, please refer to the project homepage or open an issue on the repository.












.. _installation:


############
Installation
############


Get started with OmniGenBench by installing it in your preferred environment. We recommend using a Python virtual environment to manage dependencies. Python 3.10 or higher is required.


The simplest way to install OmniGenBench is via ``pip``. This will install the latest stable release along with its core dependencies.

**************************
Installation with pip
**************************

We recommend installing OmniGenBench in a virtual environment. If you're not familiar with Python virtual environments, check out this `guide <https://docs.python.org/3/guide/ecl.html#virtual-environments-and-packages>`_.

Create a virtual environment and activate it:

.. code-block:: bash

   python3 -m venv .env
   source .env/bin/activate

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
   git clone https://github.com/yangheng95/OmniGenBench.git
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



.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: Get Started

..    installation

.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: Core Usage Guide

..    usage

.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: Command Usage Examples

..    cli

.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: Package Design Principles

..    design_principle

.. .. toctree::
..    :maxdepth: 1
..    :hidden:
..    :caption: API Reference

..    api_reference