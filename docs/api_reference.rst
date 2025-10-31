API Reference
=============

OmniGenBench is a comprehensive toolkit for genomic foundation models. It provides automated benchmarking and training pipelines, a hub for accessing pre-trained models, datasets, and pipelines, and a flexible framework for building custom models and tasks.

.. tip::

    **How to Use This Reference:**

    - Browse API documentation by functional category below.
    - Each section contains detailed classes, functions, usage examples, and extension guidelines.
    - Start with **Abstract Classes** to understand the core interfaces (OmniModel, OmniDataset, OmniTokenizer, OmniMetric).
    - Explore **Auto-Pipelines**, **CLI Commands**, and **Online Hubs** for high-level workflows.
    - Dive into **Low-level Modules** for detailed implementation and customization.

    **Quick Links:**
    
    - :doc:`api/trainers` - Complete trainer documentation (Native, Accelerate, HuggingFace)
    - :doc:`api/downstream_datasets` - Dataset classes and usage
    - :doc:`api/downstream_models` - Model architectures
    - :doc:`api/commands` - CLI command reference

**Main API Categories:**

- **Abstract Classes:**  Core interfaces (OmniModel, OmniDataset, OmniTokenizer, OmniMetric) - use as base classes for extensions
- **Auto-Pipelines:**  Automated benchmarking (AutoBench) and training (AutoTrain) workflows
- **CLI Commands:**  Command-line tools for codeless operations (ogb, autobench, autotrain, autoinfer)
- **Online Hubs:**  Access pre-trained models, datasets, and pipelines from HuggingFace Hub
- **Low-level Modules:**  Detailed modules for models, datasets, metrics, tokenizers, trainers, and utilities

Refer to each section below for detailed documentation and usage instructions.

.. Abstract Classes
.. ----------------

.. toctree::
    :maxdepth: 1
    :hidden:

    api/abstract_class

.. Auto-Pipelines
.. --------------

.. toctree::
    :maxdepth: 1
    :hidden:

    api/auto


.. CLI Commands
.. ------------

.. toctree::
    :maxdepth: 1
    :hidden:

    api/commands


.. Online Hubs
.. -----------

.. toctree::
    :maxdepth: 1
    :hidden:

    api/online_hub


.. Low-level Modules
.. -----------------

.. toctree::
    :maxdepth: 1
    :hidden:

    api/modules