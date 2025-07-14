Low-level Modules
=================

This section documents the low-level, fine-grained modules of OmniGenBench, including models, datasets, metrics, tokenizers, and trainers. These modules provide the building blocks for custom development and advanced usage.

**Overview:**

- Low-level modules expose the core functionality and implementation details of OmniGenBench.
- Users can explore these modules to understand the internal design, extend functionality, or troubleshoot specific issues.
- Each module is documented with its classes, functions, and usage examples.

**Categories:**

- Models: Classification, regression, and specialized genomic models.
- Datasets: Classes for handling various genomic data formats.
- Metrics: Evaluation metrics for classification, regression, and ranking tasks.
- Tokenizers: Sequence representation and preprocessing tools.
- Trainers: Training utilities for different frameworks and workflows.

Refer to the documentation below for details on each module, including available classes, methods, and extension options.

Models for Downstream Genomic Tasks
-----------------------------------

Classification Models
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.classification.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Regression Models
~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.regression.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

.. automodule:: omnigenome.src.model.regression.resnet
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Embedding Models
~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.embedding.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

MLM Models
~~~~~~~~~~

.. automodule:: omnigenome.src.model.mlm.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

RNA Design Models
~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.rna_design.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Sequence-to-Sequence Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.seq2seq.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Augmentation Models
~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.augmentation.model
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Model Utilities
~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.model.module_utils
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Metrics
-------

Base Metrics
~~~~~~~~~~~~

.. automodule:: omnigenome.src.metric.metric
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.metric.classification_metric
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Regression Metrics
~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.metric.regression_metric
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Ranking Metrics
~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.metric.ranking_metric
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Tokenizers
----------

BPE Tokenizer
~~~~~~~~~~~~~

.. automodule:: omnigenome.src.tokenizer.bpe_tokenizer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

K-mers Tokenizer
~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.tokenizer.kmers_tokenizer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Single Nucleotide Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.tokenizer.single_nucleotide_tokenizer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Trainers
--------

Base Trainer
~~~~~~~~~~~~

.. automodule:: omnigenome.src.trainer.trainer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

HuggingFace Trainer
~~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.trainer.hf_trainer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Accelerate Trainer
~~~~~~~~~~~~~~~~~~

.. automodule:: omnigenome.src.trainer.accelerate_trainer
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:

Miscellaneous
-------------

Utilities
~~~~~~~~~

.. automodule:: omnigenome.src.misc.utils
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :noindex:
