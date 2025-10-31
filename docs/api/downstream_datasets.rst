Downstream Datasets
----------------------------

.. note::
   
   **You are viewing the API reference documentation.**
   
   This page provides detailed API documentation for dataset classes. For a comprehensive feature guide with complete examples, see the :doc:`../OMNIDATASET_FEATURES`.
   
   - **Quick Reference**: See below for class signatures
   - **Complete Guide**: :doc:`../OMNIDATASET_FEATURES` (87KB guide with 9 sections)
   - **Design Philosophy**: :doc:`../design_principle` (understanding OmniDataset abstraction)

This module provides templated dataset processing classes inherited from the abstract ``OmniDataset`` class, which handles datasets in the OmniGenBench framework.

**Dataset Categories:**

- **OmniDatasetForSequenceClassification**: Sequence classification tasks (e.g., promoter prediction)
- **OmniDatasetForRegression**: Sequence regression tasks (e.g., translation efficiency)
- **OmniDatasetForTokenClassification**: Token (nucleotide) classification tasks (e.g., TFB prediction)
- **OmniDatasetForTokenRegression**: Token (nucleotide) regression tasks
- **OmniDatasetForMultiLabelClassification**: Multi-label classification tasks

OmniDataset
~~~~~~~~~~~~~~~~~

.. automodule:: omnigenbench.src.dataset.omni_dataset
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex: