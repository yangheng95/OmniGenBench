Abstract Classes
================

This page explains how OmniGenomeBench organizes its main concepts and how to use abstract classes to extend functionality.

OmniGenomeBench provides a set of abstract base classes that define the core interfaces for datasets, models, metrics, and tokenizers. These abstract classes are designed to be subclassed, allowing users to implement custom logic for new data formats, model architectures, evaluation metrics, or sequence representations.

**How to Use Abstract Classes:**
- Start by exploring the abstract base classes for datasets, models, metrics, and tokenizers.
- To add new functionality, subclass the relevant abstract class and implement the required methods.
- The package uses these abstract classes as the foundation for all built-in and user-extended components, ensuring consistency and interoperability.

**Main Abstract Classes:**
- `OmniDataset`: Base class for datasets. Subclass to support new data formats or preprocessing logic.
- `OmniModel`: Base class for models. Subclass to implement custom architectures or tasks.
- `OmniMetric`: Base class for evaluation metrics. Subclass to define new metrics for benchmarking.
- `OmniTokenizer`: Base class for tokenizers. Subclass to support new sequence representations.

Refer to the API documentation below for details on each abstract class, including their methods and usage examples.

Main Package Entry Point
------------------------

.. automodule:: omnigenome
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Abstract Base Classes
---------------------

.. automodule:: omnigenome.src.abc.abstract_dataset
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.src.abc.abstract_metric
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.src.abc.abstract_model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.src.abc.abstract_tokenizer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Dataset
-------

.. automodule:: omnigenome.src.dataset.omni_dataset
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

LoRA
----

.. automodule:: omnigenome.src.lora.lora_model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:
