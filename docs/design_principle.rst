.. _design_principle:

###########################
Package Design Principles
###########################



OmniGenBench is designed to be a unified, extensible, and robust framework for genomic foundation models. Our core philosophy centers on **abstraction**, **modularity**, and **interoperability**, enabling you to build, extend, and integrate complex genomic pipelines with minimal friction.

This guide explores the core architecture, the main abstract classes, and the patterns you can follow to extend the library for your own needs.

*********************
Core Philosophy
*********************

The entire framework is built upon a set of abstract base classes (ABCs). These classes define a clear "contract" or interface for every major component. This approach provides several key advantages:

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Consistency
      :shadow: md

      All components of the same type (e.g., all models) share the same interface, making them predictable and reducing bugs.

   .. grid-item-card:: Extensibility
      :shadow: md

      Adding new functionality is as simple as subclassing an existing abstract class and implementing the required methods.

   .. grid-item-card:: Interoperability
      :shadow: md

      Because all components adhere to a standard interface, they can be easily swapped or combined, like LEGO bricks.

   .. grid-item-card:: Maintainability
      :shadow: md

      A clear and consistent structure makes the codebase easier to understand, debug, and maintain over time.


***********************
The Core Components
***********************

.. OmniGenBench is built around four fundamental abstract classes. Understanding these is key to mastering the library.

.. .. design:tab-set::
..    :key: core-components

..    .. design:tab-item:: Abstract Model (`OmniModel`)
..       :sync: model-tab

..       The ``OmniModel`` class is the foundation for all models, providing a unified interface for initialization, forward passes, and inference.

..       **Key Features:**
..       *   Flexible initialization from pre-trained weights, configs, or PyTorch modules.
..       *   Automatic loss computation for various task types.
..       *   Standardized ``predict()`` and ``inference()`` methods.
..       *   Built-in support for saving and loading.

..       **Core Methods:**
..       *   ``__init__(config_or_model, tokenizer, **kwargs)``
..       *   ``forward(**inputs)``
..       *   ``predict(sequence)``
..       *   ``save_model(path)`` / ``load_model(path)``

..       **Usage Example:**
..       .. code-block:: python

..          from omnigenbench import OmniModelForSequenceClassification

..          model = OmniModelForSequenceClassification("model_path", tokenizer)
..          # Training: forward pass with labels
..          outputs = model(input_ids=..., attention_mask=..., labels=...)
..          loss = outputs.loss
..          # Inference
..          predictions = model.predict("ACGU...")
..          print(predictions)


..    .. design:tab-item:: Abstract Dataset (`OmniDataset`)
..       :sync: dataset-tab

..       The ``OmniDataset`` class standardizes data handling, supporting various file formats and integrating seamlessly with tokenizers and PyTorch DataLoaders.

..       **Key Features:**
..       *   Handles multiple data formats (JSON, CSV, Parquet, TXT).
..       *   Integrates tokenization directly into the data loading pipeline.
..       *   Automatic mapping between string labels and integer indices.
..       *   Built-in data validation and flexible configuration.

..       **Core Methods:**
..       *   ``__init__(data_path, tokenizer, **kwargs)``
..       *   ``__getitem__(index)`` & ``__len__()``
..       *   ``get_labels()``
..       *   ``get_label_mapping()``

..       **Usage Example:**
..       .. code-block:: python

..          from omnigenbench import OmniDatasetForSequenceClassification

..          dataset = OmniDatasetForSequenceClassification("data.json", tokenizer, max_length=512)
..          # Access a sample
..          sample = dataset[0]
..          print(sample['input_ids'].shape) # torch.Size([512])
..          # Get dataset info
..          print(f"Dataset size: {len(dataset)}")


..    .. design:tab-item:: Abstract Tokenizer (`OmniTokenizer`)
..       :sync: tokenizer-tab

..       The ``OmniTokenizer`` class provides a consistent wrapper for various tokenization strategies, from simple k-mers to complex pre-trained tokenizers.

..       **Key Features:**
..       *   Consistent API regardless of the underlying tokenization logic.
..       *   Automatic handling of special tokens (BOS, EOS, PAD).
..       *   Built-in preprocessing options (e.g., U-to-T conversion).
..       *   Easy integration with custom tokenization logic.

..       **Core Methods:**
..       *   ``__init__(base_tokenizer, **kwargs)``
..       *   ``tokenize(sequence, **kwargs)``
..       *   ``encode(sequence, **kwargs)`` & ``decode(token_ids, **kwargs)``
..       *   ``from_pretrained(model_name)``

..       **Usage Example:**
..       .. code-block:: python

..          from omnigenbench import OmniSingleNucleotideTokenizer

..          tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
..          # Tokenize a sequence
..          inputs = tokenizer("ATCG", max_length=128, padding=True)
..          print(inputs['input_ids'].shape)
..          # Decode back to string
..          decoded = tokenizer.decode(inputs['input_ids'][0])


..    .. design:tab-item:: Abstract Metric (`OmniMetric`)
..       :sync: metric-tab

..       The ``OmniMetric`` class standardizes evaluation, leveraging powerful libraries like `scikit-learn` while providing a consistent interface.

..       **Key Features:**
..       *   Seamless integration with `scikit-learn`'s metric collection.
..       *   Proper handling of ignored labels (e.g., -100 in PyTorch).
..       *   Standardized result dictionary format.
..       *   Support for classification, regression, and ranking metrics.

..       **Core Methods:**
..       *   ``__init__(ignore_y=None, **kwargs)``
..       *   ``compute_metric(y_true, y_pred, **kwargs)``
..       *   ``get_metric_name()``

..       **Usage Example:**
..       .. code-block:: python

..          from omnigenbench import ClassificationMetric

..          metric = ClassificationMetric(ignore_y=-100)
..          y_true = [0, 1, -100, 1]
..          y_pred = [0, 1, 0, 0]
..          results = metric.compute_metric(y_true, y_pred)
..          print(results) # {'accuracy_score': 0.66, ...}


OmniGenBench is built around four fundamental abstract classes. Understanding these is key to mastering the library.

.. card:: Abstract Model (`OmniModel`)
   :shadow: md

   The ``OmniModel`` class is the foundation for all models, providing a unified interface for initialization, forward passes, and inference.

   **Key Features:**

   *   Flexible initialization from pre-trained weights, configs, or PyTorch modules.
   *   Automatic loss computation for various task types.
   *   Standardized ``predict()`` and ``inference()`` methods.
   *   Built-in support for saving and loading.

   **Core Methods:**

   *   ``__init__(config_or_model, tokenizer, **kwargs)``
   *   ``forward(**inputs)``
   *   ``predict(sequence)``
   *   ``save_model(path)`` / ``load_model(path)``

   **Usage Example:**

   .. code-block:: python

      from omnigenbench import OmniModelForSequenceClassification

      model = OmniModelForSequenceClassification("model_path", tokenizer)
      # Training: forward pass with labels
      outputs = model(input_ids=..., attention_mask=..., labels=...)
      loss = outputs.loss
      # Inference
      predictions = model.predict("ACGU...")
      print(predictions)


.. card:: Abstract Dataset (`OmniDataset`)
   :shadow: md

   The ``OmniDataset`` class standardizes data handling, supporting various file formats and integrating seamlessly with tokenizers and PyTorch DataLoaders.

   **Key Features:**

   *   Handles multiple data formats (JSON, CSV, Parquet, TXT).
   *   Integrates tokenization directly into the data loading pipeline.
   *   Automatic mapping between string labels and integer indices.
   *   Built-in data validation and flexible configuration.

   **Core Methods:**
   
   *   ``__init__(data_path, tokenizer, **kwargs)``
   *   ``__getitem__(index)`` & ``__len__()``
   *   ``get_labels()``
   *   ``get_label_mapping()``

   **Usage Example:**

   .. code-block:: python

      from omnigenbench import OmniDatasetForSequenceClassification

      dataset = OmniDatasetForSequenceClassification("data.json", tokenizer, max_length=512)
      # Access a sample
      sample = dataset[0]
      print(sample['input_ids'].shape) # torch.Size([512])
      # Get dataset info
      print(f"Dataset size: {len(dataset)}")


.. card:: Abstract Tokenizer (`OmniTokenizer`)
   :shadow: md

   The ``OmniTokenizer`` class provides a consistent wrapper for various tokenization strategies, from simple k-mers to complex pre-trained tokenizers.

   **Key Features:**
   
   *   Consistent API regardless of the underlying tokenization logic.
   *   Automatic handling of special tokens (BOS, EOS, PAD).
   *   Built-in preprocessing options (e.g., U-to-T conversion).
   *   Easy integration with custom tokenization logic.

   **Core Methods:**

   *   ``__init__(base_tokenizer, **kwargs)``
   *   ``tokenize(sequence, **kwargs)``
   *   ``encode(sequence, **kwargs)`` & ``decode(token_ids, **kwargs)``
   *   ``from_pretrained(model_name)``

   **Usage Example:**

   .. code-block:: python

      from omnigenbench import OmniSingleNucleotideTokenizer

      tokenizer = OmniSingleNucleotideTokenizer.from_pretrained("model_name")
      # Tokenize a sequence
      inputs = tokenizer("ATCG", max_length=128, padding=True)
      print(inputs['input_ids'].shape)
      # Decode back to string
      decoded = tokenizer.decode(inputs['input_ids'][0])


.. card:: Abstract Metric (`OmniMetric`)
   :shadow: md

   The ``OmniMetric`` class standardizes evaluation, leveraging powerful libraries like `scikit-learn` while providing a consistent interface.

   **Key Features:**

   *   Seamless integration with `scikit-learn`'s metric collection.
   *   Proper handling of ignored labels (e.g., -100 in PyTorch).
   *   Standardized result dictionary format.
   *   Support for classification, regression, and ranking metrics.

   **Core Methods:**

   *   ``__init__(ignore_y=None, **kwargs)``
   *   ``compute_metric(y_true, y_pred, **kwargs)``
   *   ``get_metric_name()``

   **Usage Example:**
   
   .. code-block:: python

      from omnigenbench import ClassificationMetric

      metric = ClassificationMetric(ignore_y=-100)
      y_true = [0, 1, -100, 1]
      y_pred = [0, 1, 0, 0]
      results = metric.compute_metric(y_true, y_pred)
      print(results) # {'accuracy_score': 0.66, ...}


**********************************
Extending OmniGenBench: A How-To
**********************************

The true power of OmniGenBench lies in its extensibility. To add a custom component, you simply inherit from one of the core abstract classes and implement the required methods.

Below are implementation patterns for each component type.

.. .. grid:: 2 2 2 2
..    :gutter: 3

..    .. grid-item-card:: Custom Model
..       :shadow: md

..       Inherit from ``OmniModel`` and override the ``forward`` method to add your custom layers or logic.

..       .. code-block:: python

..          from omnigenbench import OmniModel
..          import torch

..          class CustomModel(OmniModel):
..              def __init__(self, config, tok, **kw):
..                  super().__init__(config, tok, **kw)
..                  self.classifier = torch.nn.Linear(...)

..              def forward(self, **inputs):
..                  outputs = self.base_model(**inputs)
..                  logits = self.classifier(outputs.last_hidden_state)
..                  # ... compute loss ...
..                  return loss, logits

..    .. grid-item-card:: Custom Dataset
..       :shadow: md

..       Inherit from an ``OmniDataset`` subclass and override ``_load_data`` or ``_process_data`` to handle your specific data format or structure.

..       .. code-block:: python

..          from omnigenbench import OmniDatasetForSequenceClassification

..          class CustomDataset(OmniDatasetForSequenceClassification):
..              def _load_data(self, data_path):
..                  # Your custom logic to read a file
..                  # and return a list of examples.
..                  ...
..                  return processed_data

..    .. grid-item-card:: Custom Tokenizer
..       :shadow: md

..       Inherit from ``OmniTokenizer`` and implement the core ``tokenize`` method with your unique tokenization strategy.

..       .. code-block:: python

..          from omnigenbench import OmniTokenizer

..          class KmerTokenizer(OmniTokenizer):
..              def tokenize(self, seq, **kw):
..                  k = self.k
..                  return [seq[i:i+k] for i in ...]

..    .. grid-item-card:: Custom Metric
..       :shadow: md

..       Inherit from ``OmniMetric`` and implement ``compute_metric`` to calculate your custom evaluation score.

..       .. code-block:: python

..          from omnigenbench import OmniMetric
..          from your_lib import special_metric

..          class MyMetric(OmniMetric):
..              def compute_metric(self, y_true, y_pred):
..                  score = special_metric(y_true, y_pred)
..                  return {"my_special_metric": score}

.. card:: Custom Model
   :shadow: md

   Inherit from ``OmniModel`` and override the ``forward`` method to add your custom layers or logic.

   .. code-block:: python

      from omnigenbench import OmniModel
      import torch

      class CustomModel(OmniModel):
          def __init__(self, config, tok, **kw):
              super().__init__(config, tok, **kw)
              self.classifier = torch.nn.Linear(...)

          def forward(self, **inputs):
              outputs = self.base_model(**inputs)
              logits = self.classifier(outputs.last_hidden_state)
              # ... compute loss ...
              return loss, logits



.. card:: Custom Dataset
   :shadow: md

   Inherit from an ``OmniDataset`` subclass and override ``_load_data`` or ``_process_data`` to handle your specific data format or structure.

   .. code-block:: python

      from omnigenbench import OmniDatasetForSequenceClassification

      class CustomDataset(OmniDatasetForSequenceClassification):
          def _load_data(self, data_path):
              # Your custom logic to read a file
              # and return a list of examples.
              ...
              return processed_data



.. card:: Custom Tokenizer
   :shadow: md

   Inherit from ``OmniTokenizer`` and implement the core ``tokenize`` method with your unique tokenization strategy.

   .. code-block:: python

      from omnigenbench import OmniTokenizer

      class KmerTokenizer(OmniTokenizer):
          def tokenize(self, seq, **kw):
              k = self.k
              return [seq[i:i+k] for i in ...]


.. card:: Custom Metric
   :shadow: md

   Inherit from ``OmniMetric`` and implement ``compute_metric`` to calculate your custom evaluation score.

   .. code-block:: python

      from omnigenbench import OmniMetric
      from your_lib import special_metric

      class MyMetric(OmniMetric):
          def compute_metric(self, y_true, y_pred):
              score = special_metric(y_true, y_pred)
              return {"my_special_metric": score}


********************************
Best Practices for Contributors
********************************

When extending the library, please follow these guidelines to ensure your contributions are robust and align with the framework's philosophy.

1.  **Always Inherit**: Start by inheriting from the most relevant abstract base class.
2.  **Implement Abstract Methods**: Ensure all required methods from the parent class are implemented.
3.  **Document Everything**: Provide clear docstrings for your new class and its methods, including examples.
4.  **Write Unit Tests**: Every new feature should be accompanied by tests to prevent future regressions.
5.  **Follow Conventions**: Adhere to the existing coding style and design patterns for consistency.
6.  **Handle Errors Gracefully**: Provide meaningful error messages for invalid inputs or failed operations.