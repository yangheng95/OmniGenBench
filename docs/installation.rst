Installation
==================

This page describes how to install **OmniGenBench** and its dependencies.


Dependencies
------------

OmniGenBench depends on the following major Python packages:

- findfile>=2.0.0
- autocuda>=0.16
- metric-visualizer>=0.9.6
- termcolor
- gitpython
- torch>=2.6.0
- pandas
- viennarna
- scikit-learn
- accelerate
- transformers>=4.46.0
- packaging
- peft
- dill


Basic Installation
------------------

We recommend installing via `pip`:

.. code-block:: bash

   pip install omnigenbench


Development Installation
-----------------------

Python 3.10 or higher is recommended.

To install development and testing dependencies:

.. code-block:: bash

   git clone https://github.com/yangheng95/OmniGenBench.git
   cd OmniGenBench
   python setup.py install


Optional Dependencies
---------------------

- accelerate *(for distributed training/inference)*  
- sphinx_rtd_theme *(recommended for building docs locally)*

Common Issues
-------------

- If you encounter CUDA/torch-related issues, make sure the correct version of PyTorch is installed.
- For `transformers` version conflicts, consider upgrading to the latest release.
- Windows users are strongly encouraged to use Anaconda or Miniconda environments.

For additional help, please refer to the project homepage or open an issue on the repository.
