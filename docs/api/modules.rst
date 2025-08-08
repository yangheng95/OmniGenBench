Low-level Modules
==============

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


.. Downstream Models
.. -----------------

.. Classification Models
.. ~~~~~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.classification.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. Regression Models
.. ~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.regression.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. .. automodule:: omnigenbench.src.model.regression.resnet
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. Embedding Models
.. ~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.embedding.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. MLM Models
.. ~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.mlm.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. RNA Design Models
.. ~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.rna_design.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. Sequence-to-Sequence Models
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.seq2seq.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. Augmentation Models
.. ~~~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.augmentation.model
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:

.. Model Utilities
.. ~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.model.module_utils
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:



.. Downstream Datasets
.. ----------------------------

.. This module provides some templated dataset processing classes inherited from the abstract `OmniDataset` class, which is used to handle datasets in the OmniGenBench framework.

.. **Categories:**

.. - OmniDatasetForSequenceClassification: A dataset class for sequence classification tasks.
.. - OmniDatasetForRegression: A dataset class for sequence regression tasks.
.. - OmniDatasetForTokenClassification: A dataset class for token (nucleotide) classification tasks.
.. - OmniDatasetForTokenRegression: A dataset class for token (nucleotide) regression tasks.

.. OmniDataset
.. ~~~~~~~~~~~~~~~~~

.. .. automodule:: omnigenbench.src.dataset.omni_dataset
..     :members:
..     :undoc-members:
..     :show-inheritance:
..     :noindex:




.. toctree::
    :maxdepth: 2
    
    downstream_models.rst
    downstream_datasets.rst
    metrics
    tokenizers
    trainers
    miscellaneous
